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

def get_dataloaders(train_path, test_path, class_mapping, config):
        
    df_test = pd.read_csv(test_path)
    # remove the rows containing "words" in the audio_path
    df_test = fix_updrs_speech_labels(df_test)
    test_paths = df_test.audio_path.values.tolist()
    test_labels = df_test[config.training.label_key].values.tolist()
    
    df_train = pd.read_csv(train_path)
    df_train = fix_updrs_speech_labels(df_train)
    if config.training.validation.active:
        if config.training.validation.validation_type == "speaker":
            paths, speaker_ids = df_train.audio_path.values.tolist(), df_train.speaker_id.values.tolist()
            labels = df_train[config.training.label_key].values.tolist()
            unique_speaker_ids = list(set(speaker_ids))
            unique_speaker_ids.sort()
            random.shuffle(unique_speaker_ids)
            train_speaker_ids, val_speaker_ids = train_test_split(
                unique_speaker_ids, test_size=config.training.validation.validation_split, random_state=42
            )
            t_paths, t_labels, v_paths, v_labels = [], [], [], []
            for path, label, speaker_id in zip(paths, labels, speaker_ids):
                if speaker_id in train_speaker_ids:
                    t_paths.append(path)
                    t_labels.append(label)
                else:
                    v_paths.append(path)
                    v_labels.append(label)
        elif config.training.validation.validation_type == "random":
            # just 90/10 split - on paths directly
            paths, labels = df_train.audio_path.values.tolist(), df_train[config.training.label_key].values.tolist()
            t_paths, v_paths, t_labels, v_labels = train_test_split(
                paths, labels, test_size=config.training.validation.validation_split, random_state=42
            )
        else:
            raise ValueError(f"Validation is active but validation type: {config.training.validation.validation_type} is not supported")
    else:
        t_paths, t_labels = df_train.audio_path.values.tolist(), df_train[config.training.label_key].values.tolist()
        v_paths, v_labels = [], []
        
    # set model.num_classes according to the number of classes in the dataset
    config.model.num_classes = len(set(t_labels))
    
    # create datasets
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
    
    test_ds = AudioClassificationDataset(
        audio_paths=test_paths,
        labels=test_labels,
        feature_extractor_name_or_path=config.model.model_name_or_path,
        class_mapping=class_mapping,
        data_config=config.data,
        is_test=True,
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
    
    test_dl = torch.utils.data.DataLoader(
        test_ds,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
    )
    
    return train_dl, val_dl, test_dl

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

    for test_fold in range(1, config.data.num_folds+1):
        
        # info about the fold
        fold_path = config.data.fold_root_path + f"/TRAIN_TEST_{test_fold}/"
        train_path = fold_path + "train.csv"
        test_path = fold_path + "test.csv"
        train_dl, val_dl, test_dl = get_dataloaders(train_path, test_path, class_mapping, config)

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

                # save metrics per fold
                if test_fold not in results["accuracy"]:
                    results["accuracy"][test_fold] = []
                    results["precision"][test_fold] = []
                    results["recall"][test_fold] = []
                    results["f1"][test_fold] = []
                    results["roc_auc"][test_fold] = []
                    results["sensitivity"][test_fold] = []
                    results["specificity"][test_fold] = []

                results["accuracy"][test_fold].append(accuracy)
                results["precision"][test_fold].append(precision)
                results["recall"][test_fold].append(recall)
                results["f1"][test_fold].append(f1)
                results["roc_auc"][test_fold] = roc_auc
                results["sensitivity"][test_fold] = sensitivity
                results["specificity"][test_fold] = specificity

                # log metrics to comet
                if experiment is not None:
                    experiment.log_metric("training_loss_fold_" + str(test_fold), training_loss, step=epoch+1)
                    experiment.log_metric("validation_loss_fold_" + str(test_fold), val_loss, step=epoch+1)
                    experiment.log_metric("accuracy_fold_" + str(test_fold), accuracy, step=epoch+1)
                    experiment.log_metric("precision_fold_" + str(test_fold), precision, step=epoch+1)
                    experiment.log_metric("recall_fold_" + str(test_fold), recall, step=epoch+1)
                    experiment.log_metric("f1_fold_" + str(test_fold), f1, step=epoch+1)
                    experiment.log_metric("roc_auc_fold_" + str(test_fold), roc_auc, step=epoch+1)
                    experiment.log_metric("sensitivity_fold_" + str(test_fold), sensitivity, step=epoch+1)
                    experiment.log_metric("specificity_fold_" + str(test_fold), specificity, step=epoch+1)

                # save the best model
                if accuracy > best_val_accuracy:
                    print(f"Found a better model with accuracy: {accuracy:.3f} - previous best: {best_val_accuracy:.3f}")
                    best_val_accuracy = accuracy
                    # check if DataParallel
                    if isinstance(model, torch.nn.DataParallel):
                        torch.save(model.module.state_dict(), config.training.checkpoint_path + f"/fold_{test_fold}.pt")
                    else:
                        torch.save(model.state_dict(), config.training.checkpoint_path + f"/fold_{test_fold}.pt")
            else:
                # save the model
                # check if DataParallel
                if isinstance(model, torch.nn.DataParallel):
                    torch.save(model.module.state_dict(), config.training.checkpoint_path + f"/fold_{test_fold}.pt")
                else:
                    torch.save(model.state_dict(), config.training.checkpoint_path + f"/fold_{test_fold}.pt")

        # load the best model
        if isinstance(model, torch.nn.DataParallel):
            model.module.load_state_dict(torch.load(config.training.checkpoint_path + f"/fold_{test_fold}.pt"))
        else:
            model.load_state_dict(torch.load(config.training.checkpoint_path + f"/fold_{test_fold}.pt"))

        # evaluate
        test_loss, test_reference, test_predictions = eval_one_epoch(
            model=model,
            eval_dataloader=test_dl,
            device=device,
            loss_fn=loss_fn,
            is_binary_classification=is_binary_classification,
        )

        # calculate metrics
        m_dict = compute_metrics(
            test_reference, test_predictions, verbose=config.training.verbose, is_binary_classification=is_binary_classification
        )
        
        accuracy = m_dict["accuracy"]
        precision = m_dict["precision"]
        recall = m_dict["recall"]
        f1 = m_dict["f1"]
        roc_auc = m_dict["roc_auc"]
        sensitivity = m_dict["sensitivity"]
        specificity = m_dict["specificity"]
        
        test_results["accuracy"][test_fold] = accuracy
        test_results["precision"][test_fold] = precision
        test_results["recall"][test_fold] = recall
        test_results["f1"][test_fold] = f1
        test_results["roc_auc"][test_fold] = roc_auc
        test_results["sensitivity"][test_fold] = sensitivity
        test_results["specificity"][test_fold] = specificity

        print(f"Accuracy test fold {test_fold}: {accuracy:.3f}")
        print(f"Precision test fold {test_fold}: {precision:.3f}")
        print(f"Recall test fold {test_fold}: {recall:.3f}")
        print(f"F1 test fold {test_fold}: {f1:.3f}")
        print(f"ROC AUC test fold {test_fold}: {roc_auc:.3f}")
        print(f"Sensitivity test fold {test_fold}: {sensitivity:.3f}")
        print(f"Specificity test fold {test_fold}: {specificity:.3f}")
        print(f"-" * 50)

        # log metrics to comet
        if experiment is not None:
            experiment.log_metric("test_loss_fold_" + str(test_fold), test_loss)
            experiment.log_metric("test_accuracy_fold_" + str(test_fold), accuracy)
            experiment.log_metric("test_precision_fold_" + str(test_fold), precision)
            experiment.log_metric("test_recall_fold_" + str(test_fold), recall)
            experiment.log_metric("test_f1_fold_" + str(test_fold), f1)
            experiment.log_metric("test_roc_auc_fold_" + str(test_fold), roc_auc)
            experiment.log_metric("test_sensitivity_fold_" + str(test_fold), sensitivity)
            experiment.log_metric("test_specificity_fold_" + str(test_fold), specificity)

    # save results
    results_df = pd.DataFrame(test_results)
    results_df.to_csv(config.training.checkpoint_path + "/test_results.csv", index=False)
    
    fw = open(config.training.checkpoint_path + "/test_results.txt", "w")
    
    # print average of each metric (column)
    for metric in results_df.columns:
        mean_metric = results_df[metric].mean()
        std_metric = results_df[metric].std()
        print(f"{metric}: {mean_metric*100:.2f} +/- {std_metric*100:.3f}")
        fw.write(f"{metric}: {mean_metric*100:.2f} +/- {std_metric*100:.3f}\n")
        
        if experiment is not None:
            experiment.log_metric("mean_" + metric, mean_metric)
            experiment.log_metric("std_" + metric, std_metric)
            
    fw.close()