import os
from typing import List, Union, Dict
import datasets
import numpy as np
import pandas as pd
import torch
import transformers
from hipe_commons.helpers.tsv import get_tsv_data
from torch.utils.data import DataLoader, SequentialSampler

import transformers_baseline.data_preparation
from transformers_baseline.model import predict, predict_batches
from transformers_baseline.utils import get_custom_logger

logger = get_custom_logger(__name__)


def seqeval_evaluation(predictions: List[List[str]], groundtruth: List[List[str]]):
    """Simple wrapper around seqeval."""
    metric = datasets.load_metric("seqeval")
    return metric.compute(predictions=predictions, references=groundtruth)


def evaluate_dataset(dataset: transformers_baseline.data_preparation.HipeDataset,
                     model: transformers.PreTrainedModel,
                     batch_size: int,
                     device: torch.device,
                     ids_to_labels: Dict[int, str],
                     do_debug: bool = False):
    """Evaluate an entire dataset using seqeval. Is used during the main train loop.

    :param dataset: A `HipeDataset` object.
    :param model: Self explanatory
    :param batch_size: Self explanatory
    :param device: Self explanatory
    :param ids_to_labels: a dict mapping the label numbers (int) used by the model to the original
     label names (str), e.g. `{0: "O", 1: "B-PERS", ...}`
    :param do_debug: Breaks the eval loop after the first batch.
    """

    dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=batch_size)

    logger.info('Running evaluation')
    predictions = None

    for batch in dataloader:
        if predictions is None:
            predictions = predict(batch, model, device)
            groundtruth = batch["labels"].numpy()
        else:
            predictions = np.append(predictions, predict(batch, model, device), axis=0)
            groundtruth = np.append(groundtruth, batch["labels"].numpy(), axis=0)
        if do_debug:
            break

    # Remove ignored index (special tokens)
    predictions = [
        [ids_to_labels[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, groundtruth)
    ]
    groundtruth = [
        [ids_to_labels[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, groundtruth)
    ]

    return seqeval_evaluation(predictions, groundtruth)


def write_predictions_to_tsv(words: List[List[Union[str, None]]],
                             labels: List[List[Union[str, None]]],
                             tsv_line_numbers: List[List[Union[int, None]]],
                             output_file: str,
                             labels_column: int,
                             tsv_path: str = None,
                             tsv_url: str = None, ):
    """Get the source tsv, replaces its labels with predicted labels and write a new file to `output`.

    `words`, `labels` and `tsv_line_numbers` should be three alined list, so as in HipeDataset.
    """

    logger.info(f'Writing predictions to {output_file}')

    tsv_lines = [l.split('\t') for l in get_tsv_data(tsv_path, tsv_url).split('\n')]
    label_col_number = tsv_lines[0].index(labels_column)
    for i in range(len(words)):
        for j in range(len(words[i])):
            if words[i][j]:
                assert tsv_lines[tsv_line_numbers[i][j]][0] == words[i][j]
                tsv_lines[tsv_line_numbers[i][j]][label_col_number] = labels[i][j]

    with open(output_file, 'w') as f:
        f.write('\n'.join(['\t'.join(l) for l in tsv_lines]))


def evaluate_iob_files(output_dir: str, groundtruth_path: str, preds_path: str, method: str,
                       hipe_script_path: str = None, output_suffix: str = None, env: str = 'HIPE-2022-baseline'):
    """Evaluates CLEF-HIPE compliant files.
     If `method` is set to `"hipe"`, runs run CLEF-HIPE-evaluation within `os.system`. Else if `method` is set to
     `"seqeval`, imports the files as dfs."""

    if method == "hipe":
        os.system(
            f"""
            conda activate {env}; python {hipe_script_path} \
            --skip-check \
            --ref {groundtruth_path} \
            --pred {preds_path} \
            --task nerc_coarse \
            --outdir {output_dir}
            """
        )

    elif method == "seqeval":

        with open(preds_path, "r") as f:
            preds = pd.read_csv(f, delimiter="\t", skiprows=1, comment="#", usecols=[0, 1],
                                names=["TOKEN", "NE-COARSE-LIT"])

        with open(groundtruth_path, "r") as f:
            gt = pd.read_csv(f, delimiter="\t", skiprows=1, comment="#", usecols=[0, 1],
                             names=["TOKEN", "NE-COARSE-LIT"])

        # Evaluate with seqeval
        metric = datasets.load_metric("seqeval")
        results = metric.compute(predictions=[preds["NE-COARSE-LIT"].tolist()],
                                 references=[gt["NE-COARSE-LIT"].tolist()])

        results = seqeval_to_df(results)

        if output_suffix:
            results.to_csv(os.path.join(output_dir, "{}_results_{}.tsv".format(method, output_suffix)), sep="\t",
                           index=False)
        else:
            results.to_csv(os.path.join(output_dir, "{}_results.tsv".format(method)), sep="\t", index=False)


def seqeval_to_df(seqeval_output: dict, do_debug :bool= False) -> pd.DataFrame:
    """Transforms `seqeval_output` to a MultiIndex pd.DataFrame.

    :param seqeval_output: A dict containing:
        - A dict of metrics for each entity type
        - A pair "overall_metric":value for each overall metric.
        Looks like `{'ent_type1': {'precision':float, 'recall':float}, ... , 'overall_recall':float,...}

    :param do_debug: Fills empty entity types with 0.

    :return: {(ent_type,metric):value}
    """

    abbreviations = {"precision": "P", "recall": "R", "f1": "F1", "accuracy": "A", "number": "N"}
    to_df = {}
    for key in seqeval_output.keys():
        if key.startswith("overall"):

            to_df[("ALL", abbreviations[key.split("_")[1]])] = [seqeval_output[key]]
        else:
            for subkey in seqeval_output[key].keys():
                to_df[(key, abbreviations[subkey])] = [seqeval_output[key][subkey]]

    ordered_keys = [("ALL", key) for key in ["F1", "A", "P", "R"]] + \
                   [(key1, key2) for key1 in ['AAUTHOR', 'AWORK', 'FRAGREF', 'REFAUWORK', 'REFSCOPE']
                    for key2 in ['F1', 'P', 'R', 'N']]

    if do_debug:
        to_df_debug = {}
        for key in ordered_keys:
            try:
                to_df_debug[key] =to_df[key]
            except KeyError:
                to_df_debug[key] = 0
        return pd.DataFrame(to_df_debug)

    else:
        return pd.DataFrame({key :to_df[key] for key in ordered_keys})


def evaluate_hipe(dataset: 'transformers_baseline.data_preparation.HipeDataset',
                  model: transformers.PreTrainedModel,
                  device: torch.device,
                  ids_to_labels: Dict[int, str],
                  output_dir: str,
                  labels_column: str,
                  hipe_script_path: str,
                  groundtruth_tsv_path: str = None,
                  groundtruth_tsv_url: str = None,
                  batch_size: int = 8,
                  do_debug: bool = False):

    """Performs the entire pipeline to hipe-evaluate a model, i.e. :
        - Getting the model's prediction on a dataset
        - Reconstructing a HIPE-compliant tsv with these prediction
        - Comparing the predictions-tsv and the corresponding groundtruth-tsv using the hipe scorer.
        - Writing the files.


    """
    # Write tsv locally
    if groundtruth_tsv_url:
        logger.info(f'Downloading a local copy from {groundtruth_tsv_url}')
        groundtruth_tsv_path = os.path.join(output_dir, 'groundtruth.tsv')
        groundtruth_tsv_data = get_tsv_data(url=groundtruth_tsv_url)
        with open(groundtruth_tsv_path, 'w') as f:
            f.write(groundtruth_tsv_data)

    dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=batch_size)
    predictions = predict_batches(dataloader, model, device=device, do_debug=do_debug).tolist()

    # get the labels
    predictions = [
        [ids_to_labels[p] if l != -100 else None for (p, l) in zip(prediction, label)]
        for prediction, label in zip(predictions, dataset.labels)
    ]

    preds_path = os.path.join(output_dir, 'results/hipe_eval/predictions.tsv')
    write_predictions_to_tsv(dataset.words, predictions, dataset.tsv_line_numbers,
                             preds_path,
                             labels_column, groundtruth_tsv_path, groundtruth_tsv_url)

    evaluate_iob_files(output_dir=os.path.join(output_dir, 'results/hipe_eval'),
                       groundtruth_path=groundtruth_tsv_path,
                       preds_path=preds_path,
                       method='hipe',
                       hipe_script_path=hipe_script_path,
                       )


