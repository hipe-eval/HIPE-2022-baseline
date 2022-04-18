from hipe_commons.helpers.tsv import tsv_to_dict
from typing import List, Dict, Iterable, Tuple, Generator
import torch


class HipeDataset(torch.utils.data.Dataset):

    def __init__(self,
                 batch_encoding: "BatchEncoding",
                 labels: List[List[int]],
                 tsv_line_numbers: List[List[int]],
                 words: List[List[str]]):
        self.batch_encoding = batch_encoding
        self.labels = labels
        self.tsv_line_numbers = tsv_line_numbers
        self.words = words
        self.token_offsets = [e.word_ids for e in batch_encoding.encodings]

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.batch_encoding.items() if k!='overflow_to_sample_mapping'}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def recursive_iterator(iterable: Iterable, iterable_types: Tuple[Iterable] = (list, tuple)) -> Generator:
    """Iterates recursively through an iterable potentially containing iterables of `iterable_types`."""
    for x in iterable:
        if isinstance(x, iterable_types):
            for y in recursive_iterator(x):
                yield y
        else:
            yield x


def get_unique_elements(iterable: Iterable, iterable_types: Tuple[Iterable] = (list, tuple)) -> List[str]:
    """Get the list of elements from any potentially recursive iterable."""
    return list(set([l for l in recursive_iterator(iterable, iterable_types)]))


def sort_ner_labels(labels: Iterable[str]):
    """Sorts a list of CoLLN-compliant labels alphabetically, starting 'O'."""

    assert 'O' in labels, """Label 'O' not found in labels."""
    labels = [l for l in labels if l != 'O']

    uniques = set([l[2:] for l in labels])
    assert all(['B-' + u in labels and 'I-' + u in labels for u in
                uniques]), """Some labels do not have their 'B-' or 'I-' declination."""

    sorted_labels = ['O']
    for l in sorted(uniques):
        sorted_labels.append('B-' + l)
        sorted_labels.append('I-' + l)

    return sorted_labels


def align_labels(tokens_to_words_offsets: 'transformers.tokenizers.Encoding',
                 labels: List[str],
                 labels_to_ids: Dict[str, int],
                 label_all_tokens: bool = False,
                 null_label: object = -100) -> List[List[int]]:
    previous_token_index = None
    aligned_labels = []

    for token_index in tokens_to_words_offsets:
        if token_index is None:
            aligned_labels.append(null_label)

        elif token_index != previous_token_index:
            aligned_labels.append(labels_to_ids[labels[token_index]])

        else:
            if not label_all_tokens:
                aligned_labels.append(null_label)
            else:
                aligned_labels.append(
                    labels_to_ids['I' + labels[token_index][1:] if labels[token_index] != 'O' else 'O'])

        previous_token_index = token_index

    return aligned_labels


def align_elements(tokens_to_words_offsets: 'transformers.tokenizers.Encoding',
                   elements: List[str]) -> List[List[str]]:
    """Align `element` to a list of offsets, appending `None` if the offset is None."""

    previous_token_index = None
    aligned_elements = []

    for token_index in tokens_to_words_offsets:
        if token_index is None:
            aligned_elements.append(None)

        elif token_index != previous_token_index:
            aligned_elements.append(elements[token_index])

        else:
            aligned_elements.append(None)
        previous_token_index = token_index

    return aligned_elements



def prepare_datasets(config: 'argparse.Namespace', tokenizer):
    data = {}
    for split in ['train', 'eval']:
        if config.__dict__[split + '_path'] or config.__dict__[split + '_url']:
            data[split] = tsv_to_dict(path=config.__dict__[split + '_path'], url=config.__dict__[split + '_url'])

    config.unique_labels = sort_ner_labels(
        get_unique_elements([data[k][config.labels_column] for k in data.keys()]))
    config.labels_to_ids = {label: i for i, label in enumerate(config.unique_labels)}
    config.ids_to_labels = {id: tag for tag, id in config.labels_to_ids.items()}
    config.num_labels = len(config.unique_labels)


    for split in data.keys():
        data[split]['batchencoding'] = tokenizer(data[split]['TOKEN'],
                                                 padding=True,
                                                 truncation=True,
                                                 is_split_into_words=True,
                                                 return_overflowing_tokens=True)

        data[split]['labels'] = [align_labels(e.word_ids, data[split][config.labels_column], config.labels_to_ids)
                                 for e in data[split]['batchencoding'].encodings]

        data[split]['words'] = [align_elements(e.word_ids, data[split]['TOKEN']) for e in
                                data[split]['batchencoding'].encodings]

        data[split]['tsv_line_numbers'] = [align_elements(e.word_ids, data[split]['n']) for e in
                                data[split]['batchencoding'].encodings]


    datasets = {}
    for split in data.keys():
        datasets[split] = HipeDataset(data[split]['batchencoding'],
                                      data[split]['labels'],
                                      data[split]["tsv_line_numbers"],
                                      data[split]['words'])

    return datasets


