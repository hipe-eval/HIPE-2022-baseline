"""This module is highly inspired by HuggingFace's 
[`run_ner.py`](https://github.com/huggingface/transformers/blob/main/examples/pytorch/token-classification/run_ner.py). 
It runs on a single GPU."""

import json
import logging
import time
from typing import Optional

import pandas as pd
import os
import numpy as np
import torch
import transformers
from torch.utils.data import DataLoader, RandomSampler
from transformers_baseline.evaluation import evaluate_dataset, seqeval_to_df, evaluate_hipe
from transformers_baseline.config import initialize_config
from transformers_baseline.data_preparation import prepare_datasets
from transformers_baseline.model import predict_and_write_tsv
from transformers_baseline.utils import set_seed, get_custom_logger


def train(config: 'argparse.Namespace',
          model: transformers.PreTrainedModel,
          train_dataset: 'transformers_baseline.data_preparation.HipeDataset',
          eval_dataset: Optional['transformers_baseline.data_preparation.HipeDataset'],
          tokenizer: transformers.PreTrainedTokenizer):
    """
    Main function of the the script :
        - Does the training on `train_dataset`
        - At the end of each epoch :
            - evaluate the model on `eval_dataset` using seqeval
            - saves a model checkpoint
            - saves the model as best_model if model has highest scores

    :param config: An `argparse.Namespace` containing the required `transformers.TrainingArguments`
    :param model: A transformers model instance
    :param train_dataset: The `torch.utils.data.Dataset` on which to train
    :param eval_dataset: The `torch.utils.data.Dataset` on which to evaluate the model
    :param tokenizer: The model's tokenizer.
    """

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=config.batch_size)

    t_total = len(train_dataloader) // config.gradient_accumulation_steps * config.epochs

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": config.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=config.learning_rate, eps=config.adam_epsilon)
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=config.warmup_steps, num_training_steps=t_total
    )
    # =================== Pretraining declarations ====================
    logger.info(f"""Running training on {len(train_dataset)} examples, for {config.epochs} epochs.""")

    global_step = 0
    best_f1 = 0
    count_no_improvement = 0
    train_results = pd.DataFrame()

    model.zero_grad()
    set_seed(config.seed)

    for epoch_num in range(config.epochs):

        logger.info(f"Starting epoch {epoch_num}")

        loss_batches_list = []
        epoch_time = time.time()

        for step, batch in enumerate(train_dataloader):

            model.train()
            inputs = {key: batch[key].to(config.device) for key in batch.keys()}
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are tuples in pytorch and transformers

            if config.gradient_accumulation_steps > 1:
                loss = loss / config.gradient_accumulation_steps

            loss.backward()
            loss_batches_list.append(loss.item())

            if (step + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if config.do_debug:
                    break

        # ============================ Evaluate and append during training ============================
        if config.evaluate_during_training:
            epoch_results = evaluate_dataset(eval_dataset, model, config.batch_size, config.device,
                                             config.ids_to_labels, config.do_debug)
            epoch_results = seqeval_to_df(epoch_results)

            epoch_data = pd.DataFrame({("TRAINING", "EP"): [epoch_num + 1],
                                       ("TRAINING", "TIME"): [time.time() - epoch_time],
                                       ("TRAINING", "LOSS"): [np.mean(loss_batches_list)]})
            epoch_results = pd.concat([epoch_data, epoch_results], axis=1)
            train_results = pd.concat([train_results, epoch_results], axis=0, ignore_index=True)

            # ========================= Save best model and write its results ===============================
            if round(epoch_results[('ALL', 'F1')][0], 4) > round(best_f1, 4):

                count_no_improvement = 0
                best_f1 = epoch_results[('ALL', 'F1')][0]
                epoch_results.to_csv(os.path.join(config.seqeval_output_dir, "best_results.tsv"), sep='\t', index=False)

                model.save_pretrained(config.model_save_dir)
                tokenizer.save_pretrained(config.model_save_dir)
                torch.save(config, os.path.join(config.model_save_dir, "training_args.bin"))


            else:
                count_no_improvement += 1

            if count_no_improvement == config.early_stopping_patience and config.do_early_stopping:
                break

        if config.do_debug:
            break
    if config.evaluate_during_training:
        train_results.to_csv(os.path.join(config.seqeval_output_dir, "train_results.tsv"), sep='\t', index=False)
    else:
        model.save_pretrained(config.model_save_dir)
        tokenizer.save_pretrained(config.model_save_dir)
        torch.save(config, os.path.join(config.model_save_dir, "training_args.bin"))


def main(config):
    logger.info(f'Runing pipeline on {config.output_dir.split("/")[-1]}')

    # Create directories
    os.makedirs(config.output_dir, exist_ok=config.overwrite_output_dir)

    config.model_save_dir = os.path.join(config.output_dir, "model")
    os.makedirs(config.model_save_dir, exist_ok=config.overwrite_output_dir)

    config.predictions_dir = os.path.join(config.output_dir, "predictions")
    os.makedirs(config.predictions_dir, exist_ok=config.overwrite_output_dir)

    config.seqeval_output_dir = os.path.join(config.output_dir, 'results/seqeval')
    os.makedirs(config.seqeval_output_dir, exist_ok=config.overwrite_output_dir)

    config.hipe_output_dir = os.path.join(config.output_dir, 'results/hipe_eval')
    os.makedirs(config.hipe_output_dir, exist_ok=config.overwrite_output_dir)

    # Save config
    with open(os.path.join(config.output_dir, 'config.json'), 'w') as f:
        json.dump(config.__dict__, f, skipkeys=True, indent=4, sort_keys=True,
                  default=lambda o: '<not serializable>')

    # todo change model_name_or_path to model_config ; make a double path on data
    # tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_name_or_path, add_prefix_space=True)  # for roberta exclusively
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_name_or_path)

    datasets = prepare_datasets(config, tokenizer)

    model = transformers.AutoModelForTokenClassification.from_pretrained(config.model_name_or_path,
                                                                    num_labels=config.num_labels
                                                                         )
    model.to(config.device)

    if config.do_train:
        train(config=config,
              model=model,
              train_dataset=datasets['train'],
              eval_dataset=datasets['eval'] if config.evaluate_during_training else None,
              tokenizer=tokenizer)

    if config.do_hipe_eval:
        evaluate_hipe(dataset=datasets['eval'],
                      model=model,
                      device=config.device,
                      ids_to_labels=config.ids_to_labels,
                      output_dir=config.hipe_output_dir,
                      labels_column=config.labels_column,
                      hipe_script_path=config.hipe_script_path,
                      groundtruth_tsv_path=config.eval_path,
                      groundtruth_tsv_url=config.eval_url,
                      )

    if config.do_predict:
        for url in config.predict_urls:
            predict_and_write_tsv(model=model, device=config.device, output_dir=config.predictions_dir,
                                  tokenizer=tokenizer, ids_to_labels=config.ids_to_labels,
                                  labels_column=config.labels_column, url=url)

        for path in config.predict_paths:
            predict_and_write_tsv(model=model, device=config.device, output_dir=config.predictions_dir,
                                  tokenizer=tokenizer, ids_to_labels=config.ids_to_labels,
                                  labels_column=config.labels_column, url=path)




if __name__ == '__main__':
    logger = get_custom_logger(__name__, level=logging.DEBUG)
    config = initialize_config(
        # json_path='configs/newseye_de_coarse.json'
    )
    main(config)
