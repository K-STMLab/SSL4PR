# SSL4PR: Self-supervised learning for Parkinson's Recognition

This project aims at creating a DL system for Parkinson's recognition from speech. It leverages self-supervised learning models to transfer knowledge acquired by foundational models to the task of Parkinson's recognition. The project is based on the PC-GITA dataset and the proposed model is evaluated using 10-fold cross-validation.

**Table of Contents**
- [Setup](#setup)
- [Dataset and Data Splits](#dataset-and-data-splits)
- [Experiments](#experiments)
- [Citation](#citation)

### Setup

The project is based on Python 3.11 and PyTorch 2. The following command can be used to install the dependencies:

```bash
pip install -r requirements.txt
```

By default, the project leverages `comet_ml` for logging. To use it, you need to create an account on [comet.ml](https://www.comet.ml/). Then, you need to set the following environment variables:

```bash
export COMET_API_KEY=<your_api_key>
export COMET_WORKSPACE=<your_project_name>
```

**Disable logging**: 
To disable the logging, you can set the `training.use_comet` key to `false` in the `configs/*.yaml` files.

### Dataset and Data Splits

To make the results reproducible and comparable with the ones reported in the paper we make available the data splits used for 10-fold cross-validation. The splits are available in the `pcgita_splits` folder. The data is organized as follows:

```
pcgita_splits/
├── TRAIN_TEST_1
│   ├── test.csv
│   └── train.csv
├── TRAIN_TEST_2
│   ├── test.csv
│   └── train.csv
├── ...
└── TRAIN_TEST_10
    ├── test.csv
    └── train.csv
```

The `train.csv` and `test.csv` files contain the list of the audio files used for training and testing, respectively. The path to the audio files is stored in the following format:

```
/PC_GITA_ROOT_PATH/monologue/sin_normalizar/pd/AVPEPUDEA0042-Monologo-NR.wav
```

where `PC_GITA_ROOT_PATH` is the root path to the PC-GITA dataset. We also provide a python script `set_root_path.py` to set the root path to the PC-GITA dataset. The script can be used as follows:

```bash
python set_root_path.py --old_root_path <old_root_path> --new_root_path <new_root_path>
```

where `<old_root_path>` can be set to `PC_GITA_ROOT_PATH` and `<new_root_path>` is the new root path to your instance of the PC-GITA dataset.

**Note**: The splits are generated to ensure that the same speaker does not appear in both the training and testing sets and the classes are balanced across the splits.

### Experiments

To train the proposed model it is first needed to install the requirements and set the root path to the PC-GITA dataset. Then, the following command can be used to train the model:

```bash
python train.py --config <config_file> 
```

where `<config_file>` is the path to the configuration file (e.g., `configs/W_config.yaml`). There are several configuration parameter that can be set in the configuration file. The one reported in the paper is `configs/W_config.yaml`.

The training file creates 10 models, one for each fold, and compute the metrics for each fold. The metrics are reported on terminal and logged on comet.ml (if activated). Please be sure to update the following path in the configuration file:

```yaml
training:
  checkpoint_path: <path_to_save_checkpoints>
data:
  fold_root_path: <path_to_pcgita_splits>
```

where `<path_to_save_checkpoints>` is the path where the checkpoints will be saved and `<path_to_pcgita_splits>` is the path to the `pcgita_splits` folder.

#### Inference on extended dataset

The paper presents the results of the proposed model on an extended dataset. The extended dataset is available in the `extended_dataset` folder. The script `infer_extended.py` can be used to infer the model on the extended dataset. The script can be used as follows:

```bash
python infer_extended.py --config <config_file> --training.ext_model_path <path_to_model> --ext_root_path <path_to_extended_dataset>
```

where `<config_file>` is the path to the configuration file (e.g., `configs/W_config.yaml`), `<path_to_model>` is the path to the model checkpoint and `<path_to_extended_dataset>` is the path to the `extended_dataset` folder.

**Speech Enhancement**: The proposed model can be used on the extended dataset in combination with speech enhancement preprocessing. To preprocess data following the same process as in the paper, the `speech_enhancement` folder contains the instructions to apply, VAD, dereverberation and noise reduction to the audio files.

### Citation

If you use this code, results from this project or you want to refer to the paper, please cite the following paper:

```
# currently under review
```

