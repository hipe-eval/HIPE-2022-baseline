# HIPE-2022-baseline

This repository contains the baseline models for HIPE-2022. 


## `transformers_baseline`

### Presentation 
`transformers_baseline` contains the functions and config files to run a transformer-based baseline on 
on HIPE-2022 datasets. Notable functionalities include: 

- Data preparation (`data_preparation.py`):
  - Import HIPE-compliant `tsv`s, using [`hipe_commons`](https://github.com/hipe-eval/HIPE-pycommons). 
  - Tokenizing the data and aligning the labels
  - Creating custom datasets amenable to HuggingFace's transformers (see `HipeDataset`)

- Training and fine-tuning (`pipeline.py`)
  - Single GPU support, based on HuggingFace [`run_ner.py`](https://github.com/huggingface/transformers/blob/main/examples/pytorch/token-classification/run_ner.py).
  - Evaluation during training using `seqeval`
  
- Evaluation (`evaluation.py`)
  - Predict functions
  - Reconstruct HIPE-compliant tsv from the model's predictions
  - Evaluate `tsv`s using [`HIPE-scorer`](https://github.com/hipe-eval/HIPE-scorer).

### Usage

Make sure you have the packages listed in `requirements.txt` installed in a virtual environment. Then please run : 

```shell
python pipeline.py
```

With the following args:
```
usage: pipeline.py [-h] [--train_path TRAIN_PATH] [--train_url TRAIN_URL] [--eval_path EVAL_PATH] [--eval_url EVAL_URL] [--output_dir OUTPUT_DIR]
                   [--hipe_script_path HIPE_SCRIPT_PATH] [--config_path CONFIG_PATH] [--labels_column LABELS_COLUMN] [--text_column_name TEXT_COLUMN_NAME]
                   [--segmentation_flag SEGMENTATION_FLAG] [--model_name_or_path MODEL_NAME_OR_PATH] [--do_train] [--do_hipe_eval] [--do_debug] [--overwrite_output_dir]
                   [--device_name DEVICE_NAME] [--epochs EPOCHS] [--seed SEED] [--batch_size BATCH_SIZE] [--gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS]

options:
  -h, --help            show this help message and exit
  --train_path TRAIN_PATH
                        Absolute path to the tsv data file to train on
  --train_url TRAIN_URL
                        url to the tsv data file to train on
  --eval_path EVAL_PATH
                        Absolute path to the tsv data file to evaluate on
  --eval_url EVAL_URL   url to the tsv data file to evaluate on
  --output_dir OUTPUT_DIR
                        Absolute path to the directory in which outputs are to be stored
  --hipe_script_path HIPE_SCRIPT_PATH
                        The path the CLEF-HIPE-evaluation script. This parameter is required if `do_hipe_eval`is True
  --config_path CONFIG_PATH
                        The path to a config json file from which to extract config. Overwrites other specified config
  --labels_column LABELS_COLUMN
                        Name of the tsv col to extract labels from
  --text_column_name TEXT_COLUMN_NAME
                        Name of the tsv col to extract texts from
  --segmentation_flag SEGMENTATION_FLAG
                        Use if you want to pre-segment data. Use 'NOSEGMENTATION' to let the tokenizer chunk the data to bits of the model's max_length (512)
  --model_name_or_path MODEL_NAME_OR_PATH
                        Absolute path to model directory or HF model name (e.g. 'bert-base-cased')
  --do_train            whether to train. Leave to false if you just want to evaluate
  --do_hipe_eval        Performs CLEF-HIPE evaluation, alone or at the end of training if `do_train`.
  --do_debug            Breaks all loops after a single iteration for debugging
  --overwrite_output_dir
                        Whether to overwrite the output dir
  --device_name DEVICE_NAME
                        Device in the format 'cuda:1', 'cpu'
  --epochs EPOCHS       Total number of training epochs to perform.
  --seed SEED           Random seed
  --batch_size BATCH_SIZE
                        Batch size per device.
  --gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS
                        Number of steps to accumulate before performing backpropagation.
```

Please note that the config can also read from a json : 

```shell
python pipeline.py --config_path '/abs/path/to/my_json_conf.json'
```

