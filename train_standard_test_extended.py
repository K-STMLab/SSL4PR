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

def set_all_seeds(seed):
    try:
        random.seed(seed)
    except:
        print("[RANDOM] Impossible to set seed for random - is it imported?")
        
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except:
        print("[TORCH] Impossible to set seed for torch - is it imported?")
        
    try:
        np.random.seed(seed)
    except:
        print("[NUMPY] Impossible to set seed for numpy - is it imported?")
    
    print(f"Set all seeds to {seed}")

def train_one_epoch(model, train_dataloader, optimizer, scheduler, device, loss_fn, experiment=None, fold_num=0, gradient_accumulation_steps=1, is_binary_classification=False):
    model.train()

    p_bar = tqdm(train_dataloader, total=len(train_dataloader), ncols=100)
    training_loss = 0.0
    log_each = 50

    for batch in p_bar:
        
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["labels"]
        outputs = model(batch)

        # skip if pred is nan
        if torch.isnan(outputs).any():
            print("Skipping batch because of nan")
            # clear memory
            del batch
            del labels
            del outputs
            torch.cuda.empty_cache()
            continue
        
        n_classes = outputs.shape[-1]

        if is_binary_classification: loss = loss_fn(outputs.squeeze(-1), labels)
        else: loss = loss_fn(outputs.view(-1, n_classes), labels.view(-1))

        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps
            
        loss.backward()
        if (p_bar.n + 1) % gradient_accumulation_steps == 0 or p_bar.n == len(train_dataloader) - 1:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        training_loss += loss.item()

        p_bar.set_postfix({"loss": loss.item(), "lr": scheduler.get_last_lr()[-1]})

        if experiment is not None:
            experiment.log_metric("training_loss_fold_" + str(fold_num), loss.item())
            experiment.log_metric("learning_rate_fold_" + str(fold_num), scheduler.get_last_lr()[-1])
    
    return training_loss / len(train_dataloader)

def eval_one_epoch(model, eval_dataloader, device, loss_fn, experiment=None, is_binary_classification=False):
    model.eval()

    p_bar = tqdm(eval_dataloader, total=len(eval_dataloader), ncols=100)
    eval_loss = 0.0
    reference = []
    predictions = []

    with torch.no_grad():
        for batch in p_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["labels"]
            
            outputs = model(batch)
            n_classes = outputs.shape[-1]

            if is_binary_classification: loss = loss_fn(outputs.squeeze(-1), labels)
            else: loss = loss_fn(outputs.view(-1, n_classes), labels.view(-1))

            eval_loss += loss.item()
            reference.extend(labels.cpu().numpy())
            if is_binary_classification: predictions.extend( (outputs > 0.5).cpu().numpy().astype(int) )
            else: predictions.extend(torch.argmax(outputs, dim=-1).cpu().numpy().astype(int))

            p_bar.set_postfix({"loss": loss.item()})

    return eval_loss / len(eval_dataloader), reference, predictions

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
        
        
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }

def get_model(config):
    model = SSLClassificationModel(config=config)
    return model

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

def get_train_dataloaders(train_path, test_path, class_mapping, config):
    df_test = pd.read_csv(test_path)
    # remove the rows containing "words" in the audio_path
    df_test = fix_updrs_speech_labels(df_test)
    test_paths = df_test.audio_path.values.tolist()
    test_labels = df_test[config.training.label_key].values.tolist()
    
    df_train = pd.read_csv(train_path)
    df_train = fix_updrs_speech_labels(df_train)
    train_paths, train_labels = df_train.audio_path.values.tolist(), df_train[config.training.label_key].values.tolist()
    
    # merge lists
    paths = train_paths + test_paths
    labels = train_labels + test_labels
    
    if config.training.validation.active:
        if config.training.validation.validation_type == "random":
            t_paths, v_paths, t_labels, v_labels = train_test_split(
                paths, labels, test_size=config.training.validation.validation_split, random_state=42
            )
        else:
            raise ValueError(f"Validation is active but validation type: {config.training.validation.validation_type} is not supported")
    else:
        t_paths, t_labels = paths, labels
        
    config.model.num_classes = len(set(t_labels))
    
    t_ds = AudioClassificationDataset(
        audio_paths=t_paths,
        labels=t_labels,
        feature_extractor_name_or_path=config.model.model_name_or_path,
        class_mapping=class_mapping,
        data_config=config.data,
    )
    
    if config.training.validation.active:
        v_ds = AudioClassificationDataset(
            audio_paths=v_paths,
            labels=v_labels,
            feature_extractor_name_or_path=config.model.model_name_or_path,
            class_mapping=class_mapping,
            data_config=config.data,
            is_test=True
        )
    
    # create dataloaders
    train_dl = torch.utils.data.DataLoader(
        t_ds,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
    )
    if config.training.validation.active:
        val_dl = torch.utils.data.DataLoader(
            v_ds,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.training.num_workers,
            pin_memory=config.training.pin_memory,
        )
    else:
        val_dl = None
    
    return train_dl, val_dl

def get_extended_test_dataloader(test_path, class_mapping, config):
    subfolders = ["DDK1" , "monologue", "readtext"]
    classes = ["HC", "PD"] 
    
    audio_paths = []
    labels = []
    
    for sf in subfolders:
        for c in classes:
            if sf == "words":
                # find another level of subfolders
                subsubfolders = os.listdir(os.path.join(test_path, sf, c))
                for ssf in subsubfolders:
                    files = os.listdir(os.path.join(test_path, sf, c, ssf))
                    for f in files:
                        audio_paths.append(os.path.join(test_path, sf, c, ssf, f))
                        labels.append(c)
            else:
                files = os.listdir(os.path.join(test_path, sf, c))
                for f in files:
                    audio_paths.append(os.path.join(test_path, sf, c, f))
                    labels.append(c)
                    
    print("Number of audio files: ", len(audio_paths))
    print("Number of labels: ", len(labels))
    
    # lowercased labels
    labels = [l.lower() for l in labels]
    
    # config.model.num_classes = len(set(labels))
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

# each fold will be used as test set - one for validation and the rest for training

if __name__ == "__main__":
    
    # ------------------------------------------
    # Setting up the training environment
    # ------------------------------------------

    config = add_arguments()
    config = Dict(config)
    set_all_seeds(config.training.seed)

    # create checkpoint path if it does not exist
    if not os.path.exists(config.training.checkpoint_path):
        os.makedirs(config.training.checkpoint_path, exist_ok=True)

    # create comet experiment if needed
    if config.training.use_comet:
        experiment = Experiment(
            api_key=os.environ["COMET_API_KEY"],
            workspace=os.environ["COMET_WORKSPACE"],
            project_name=config.training.comet_project_name,
        )
        experiment.set_name(config.training.comet_experiment_name)
        experiment.log_parameters(config)
    else:
        experiment = None


    # ------------------------------------------
    # Data preparation
    # ------------------------------------------
    if config.training.label_key == "status":
        class_mapping = {'hc':0, 'pd':1}
        is_binary_classification = True
        print(f"Class mapping: {class_mapping}")
    elif config.training.label_key == "UPDRS-speech":
        is_binary_classification = False
        class_mapping = {0: 0, 1: 1, 2: 2, 3: 3}
        print(f"Class mapping: {class_mapping}")

    results = {
        "accuracy": {},
        "precision": {},
        "recall": {},
        "f1": {},
        "roc_auc": {},
        "sensitivity": {},
        "specificity": {},
    }

    test_results = {
        "accuracy": {},
        "precision": {},
        "recall": {},
        "f1": {},
        "roc_auc": {},
        "sensitivity": {},
        "specificity": {},
    }

    # info about the fold
    fold_path = config.data.fold_root_path + f"/TRAIN_TEST_1/"
    train_path = fold_path + "train.csv"
    test_path = fold_path + "test.csv"
    train_dl, val_dl = get_train_dataloaders(train_path, test_path, class_mapping, config)
    test_dl_raw = get_extended_test_dataloader(config.training.raw_ext_root_path, class_mapping, config)
    test_dl_se = get_extended_test_dataloader(config.training.se_ext_root_path, class_mapping, config)

    # create model
    model = get_model(config)
    model, device = manage_devices(model, use_cuda=config.training.use_cuda, multi_gpu=config.training.multi_gpu)
    # print the number of parameters
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")

    # create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    # create scheduler
    total_steps = int(len(train_dl) * config.training.num_epochs) // config.training.gradient_accumulation_steps
    warmup_ratio = 0.1
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(total_steps * warmup_ratio),
        num_training_steps=total_steps,
        last_epoch=-1,
    )

    # create loss function
    if is_binary_classification: loss_fn = torch.nn.BCELoss()
    else: loss_fn = torch.nn.CrossEntropyLoss()

    print(loss_fn)
    
    # train and validate
    best_val_accuracy = 0.0
    for epoch in range(config.training.num_epochs):
        print(f"Epoch: {epoch + 1}/{config.training.num_epochs}")
        
        # train
        training_loss = train_one_epoch(
            model=model,
            train_dataloader=train_dl,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            loss_fn=loss_fn,
            is_binary_classification=is_binary_classification,
        )
        print(f"Average training loss: {training_loss}")
        
        if config.training.validation.active:
            # validate
            val_loss, val_reference, val_predictions = eval_one_epoch(
                model=model,
                eval_dataloader=val_dl,
                device=device,
                loss_fn=loss_fn,
                is_binary_classification=is_binary_classification,
            )
            print(f"Average validation loss: {val_loss}")

            # compute metrics
            m_dict = compute_metrics(
                val_reference, val_predictions, verbose=config.training.verbose, is_binary_classification=is_binary_classification
            )
            
            accuracy = m_dict["accuracy"]
            precision = m_dict["precision"]
            recall = m_dict["recall"]
            f1 = m_dict["f1"]
            roc_auc = m_dict["roc_auc"]
            sensitivity = m_dict["sensitivity"]
            specificity = m_dict["specificity"]

            # save the best model
            if accuracy > best_val_accuracy:
                print(f"Found a better model with accuracy: {accuracy:.3f} - previous best: {best_val_accuracy:.3f}")
                best_val_accuracy = accuracy
                # check if DataParallel
                if isinstance(model, torch.nn.DataParallel):
                    torch.save(model.module.state_dict(), config.training.checkpoint_path + f"/model_best.pt")
                else:
                    torch.save(model.state_dict(), config.training.checkpoint_path + f"/model_best.pt")
        else:
            # save the model
            # check if DataParallel
            if isinstance(model, torch.nn.DataParallel):
                torch.save(model.module.state_dict(), config.training.checkpoint_path + f"/model_best.pt")
            else:
                torch.save(model.state_dict(), config.training.checkpoint_path + f"/model_best.pt")

    # load the best model
    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(torch.load(config.training.checkpoint_path + f"/model_best.pt"))
    else:
        model.load_state_dict(torch.load(config.training.checkpoint_path + f"/model_best.pt"))




    # ------------------------------------------
    # Test on the extended dataset - RAW
    # ------------------------------------------
    print("Testing on the extended dataset - RAW")
    # evaluate
    test_loss, test_reference, test_predictions = eval_one_epoch(
        model=model,
        eval_dataloader=test_dl_raw,
        device=device,
        loss_fn=loss_fn,
        is_binary_classification=is_binary_classification,
    )

    # calculate metrics
    m_dict = compute_metrics(
        test_reference, test_predictions, verbose=config.training.verbose, is_binary_classification=is_binary_classification
    )
    
    fw = open(config.training.checkpoint_path + "/test_results.txt", "w")
    fw.write("********* RAW *********\n")
    print("\n\n ********* RAW *********")
    # print average of each metric (column)
    for metric in m_dict.keys():
        print(f"{metric}: {m_dict[metric]*100:.2f}")
        fw.write(f"{metric}: {m_dict[metric]*100:.2f}\n")
        
        if experiment is not None:
            experiment.log_metric(metric, m_dict[metric])
    
    
    # ------------------------------------------
    # Test on the extended dataset - SE
    # ------------------------------------------
    
    print("Testing on the extended dataset - SE")
    # evaluate
    test_loss, test_reference, test_predictions = eval_one_epoch(
        model=model,
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
    
    for metric in m_dict.keys():
        print(f"{metric}: {m_dict[metric]*100:.2f}")
        fw.write(f"{metric}: {m_dict[metric]*100:.2f}\n")
        
        if experiment is not None:
            experiment.log_metric(metric, m_dict[metric])
            
    fw.close()
    
    if experiment is not None:
        experiment.end()
        
    print("Done!")