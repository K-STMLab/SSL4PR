from comet_ml import Experiment

import os
import random
import yaml
import argparse
from tqdm import tqdm


import torch
torch.autograd.set_detect_anomaly(True)
import pandas as pd
from yaml_config_override import add_arguments

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split

from transformers import get_linear_schedule_with_warmup

from models.ssl_classification_model import SSLClassificationModel
from datasets.audio_classification_dataset import AudioClassificationDataset

from yaml_config_override import add_arguments
from addict import Dict

import numpy as np

def eval_mix_models(models, eval_dataloader, device, loss_fn, experiment=None, is_binary_classification=False):
    
    for model in models:
        model.eval()

    p_bar = tqdm(eval_dataloader, total=len(eval_dataloader), ncols=100)
    eval_loss = 0.0
    reference = []
    predictions = []

    with torch.no_grad():
        for batch in p_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["labels"]
            
            outputs = np.zeros((len(labels), 1))
            for model in models:
                outp = model(batch).detach()
                # torch.Size([32, 1])
                # sum the outputs
                outputs += outp.cpu().numpy()
            outputs /= len(models)
            n_classes = outputs.shape[-1]
            # print("outputs: ", outputs)
            reference.extend(labels.cpu().numpy())
            if is_binary_classification: predictions.extend( (outputs > 0.5).astype(int) )
            else: predictions.extend(torch.argmax(outputs, dim=-1).cpu().numpy().astype(int))

    return None, reference, predictions

def compute_metrics(reference, predictions, verbose=False, is_binary_classification=False):
    
    accuracy = accuracy_score(reference, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(reference, predictions, average="macro")
    
    if is_binary_classification:
        roc_auc = roc_auc_score(reference, predictions)
        cm = confusion_matrix(reference, predictions)
        tp = cm[1, 1]
        tn = cm[0, 0]
        fp = cm[0, 1]
        fn = cm[1, 0]
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
    else:
        print("ROC AUC is not defined for multiclass classification")
        roc_auc = 0.0
        sensitivity = 0.0
        specificity = 0.0
        
    if verbose:
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1: {f1}")
        print(f"ROC AUC: {roc_auc}")
        print(f"Sensitivity: {sensitivity}")
        print(f"Specificity: {specificity}")
        
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }

def manage_devices(model, use_cuda=True, multi_gpu=False):
    if use_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        if multi_gpu and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
    else:
        device = torch.device("cpu")
    model.to(device)
    print(f"From config: use_cuda: {use_cuda}, multi_gpu: {multi_gpu}")
    print(f"Using device: {device}")
    return model, device

def fix_updrs_speech_labels(df):
    df["UPDRS-speech"] = df["UPDRS-speech"].fillna(0)
    df["UPDRS-speech"] = df["UPDRS-speech"].astype(int)
    return df

def updrs_level_is_valid(filename):
    ext_info_path = "/mnt/disk2/mlaquatra/pc_gita_ext/info_patients.tsv"
    valid_levels = [0, 1, 2, 3]
    ext_info = pd.read_csv(ext_info_path, sep="\t")
    ext_info = ext_info.fillna(-1)
    codes = ext_info["code"].values
    updrs_speech_level = ext_info["updrs_speech"].values

    for i, code in enumerate(codes):
        if code in filename:
            print("Code: ", code, " - Filename: ", filename)
            print("updrs_speech_level[i]: ", updrs_speech_level[i])
            return updrs_speech_level[i] in valid_levels or updrs_speech_level[i] == -1
    return False

def get_extended_test_dataloader(test_path, class_mapping, config):
    subfolders = ["DDK1" , "monologue", "readtext"]
    classes = ["HC", "PD"] 
    audio_paths = []
    labels = []
    filter_updrs_level = False
    for sf in subfolders:
        for c in classes:
            if sf == "words":
                # find another level of subfolders
                subsubfolders = os.listdir(os.path.join(test_path, sf, c))
                for ssf in subsubfolders:
                    files = os.listdir(os.path.join(test_path, sf, c, ssf))
                    for f in files:
                        if filter_updrs_level: 
                            if updrs_level_is_valid(f):
                                audio_paths.append(os.path.join(test_path, sf, c, ssf, f))
                                labels.append(c)
                        else:
                            audio_paths.append(os.path.join(test_path, sf, c, ssf, f))
                            labels.append(c)
            else:
                files = os.listdir(os.path.join(test_path, sf, c))
                for f in files:
                    if filter_updrs_level:
                        if updrs_level_is_valid(f):
                            audio_paths.append(os.path.join(test_path, sf, c, f))
                            labels.append(c)
                    else:
                        audio_paths.append(os.path.join(test_path, sf, c, f))
                        labels.append(c)
                        
    print("Number of audio files: ", len(audio_paths))
    print("Number of labels: ", len(labels))
    
    print("Audio paths: ", audio_paths)
    print("Labels: ", labels)
    print("Labels per class: ", {c: labels.count(c) for c in classes})
    
    # lowercased labels
    labels = [l.lower() for l in labels]
    
    dataset = AudioClassificationDataset(
        audio_paths=audio_paths,
        labels=labels,
        feature_extractor_name_or_path=config.model.model_name_or_path,
        class_mapping=class_mapping,
        data_config=config.data,
        is_test=True,
    )
    
    dl = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
    )
    
    return dl

if __name__ == "__main__":
    
    # ------------------------------------------
    # Setting up the training environment
    # ------------------------------------------

    config = add_arguments()
    config = Dict(config)
    class_mapping = {"hc": 0, "pd": 1}
    config.model.num_classes = len(class_mapping)
    test_dl_raw = get_extended_test_dataloader(config.training.raw_ext_root_path, class_mapping, config)
    test_dl_se = get_extended_test_dataloader(config.training.se_ext_root_path, class_mapping, config)
    
    models = []

    model_1_tag = "facebook/hubert-base-ls960"
    model_1_path = "/mnt/disk2/mlaquatra/dispeeh-ckpts-mix/facebook-hubert-base-ls960/model_best.pt"
    model_2_tag = "microsoft/wavlm-base"
    model_2_path = "/mnt/disk2/mlaquatra/dispeeh-ckpts-mix/microsoft-wavlm-base/model_best.pt"
    model_3_tag = "facebook/wav2vec2-base-960h"
    model_3_path = "/mnt/disk2/mlaquatra/dispeeh-ckpts-mix/facebook-wav2vec2-base-960h/model_best.pt"
    
    config.model.model_name_or_path = model_1_tag
    model_1 = SSLClassificationModel(config=config)
    config.model.model_name_or_path = model_2_tag
    model_2 = SSLClassificationModel(config=config)
    # config.model.model_name_or_path = model_3_tag
    # model_3 = SSLClassificationModel(config=config)
    
    model_1.load_state_dict(torch.load(model_1_path))
    model_2.load_state_dict(torch.load(model_2_path))
    # model_3.load_state_dict(torch.load(model_3_path))
    
    model_1, device = manage_devices(model_1, use_cuda=True, multi_gpu=False)
    models.append(model_1)
    model_2, device = manage_devices(model_2, use_cuda=True, multi_gpu=False)
    models.append(model_2)
    # model_3, device = manage_devices(model_3, use_cuda=True, multi_gpu=False)
    # models.append(model_3)
    
    print("Models loaded")
    
    is_binary_classification = True
    loss_fn = torch.nn.BCEWithLogitsLoss() if is_binary_classification else torch.nn.CrossEntropyLoss()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # ------------------------------------------
    # Test on the extended dataset - RAW
    # ------------------------------------------
    print("Testing on the extended dataset - RAW")
    # evaluate
    test_loss, test_reference, test_predictions = eval_mix_models(
        models=models,
        eval_dataloader=test_dl_raw,
        device=device,
        loss_fn=loss_fn,
        is_binary_classification=is_binary_classification,
    )

    # calculate metrics
    m_dict = compute_metrics(
        test_reference, test_predictions, verbose=config.training.verbose, is_binary_classification=is_binary_classification
    )
    
    os.makedirs("system_combination/", exist_ok=True)
    filenamme = "system_combination/monologue_"
    for model in [model_1_tag, model_2_tag]:
        filenamme += model.split("/")[-1] + "_"
    filenamme += "metrics.txt"
    fw = open(filenamme, "w")
    
    fw.write("********* RAW *********\n")
    print("\n\n ********* RAW *********")
    # print average of each metric (column)
    for metric in m_dict.keys():
        print(f"{metric}: {m_dict[metric]*100:.2f}")
        fw.write(f"{metric}: {m_dict[metric]*100:.2f}\n")
    
    
    # ------------------------------------------
    # Test on the extended dataset - SE
    # ------------------------------------------
    
    print("Testing on the extended dataset - SE")
    # evaluate
    test_loss, test_reference, test_predictions = eval_mix_models(
        models=models,
        eval_dataloader=test_dl_se,
        device=device,
        loss_fn=loss_fn,
        is_binary_classification=is_binary_classification,
    )
    
    # calculate metrics
    m_dict = compute_metrics(
        test_reference, test_predictions, verbose=config.training.verbose, is_binary_classification=is_binary_classification
    )
    
    fw.write("********* SE *********\n")
    print("\n\n ********* SE *********")
    # print average of each metric (column)
    for metric in m_dict.keys():
        print(f"{metric}: {m_dict[metric]*100:.2f}")
        fw.write(f"{metric}: {m_dict[metric]*100:.2f}\n")
            
    fw.close()

    print("Done!")