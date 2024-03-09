import os
import torch
from yaml_config_override import add_arguments
from addict import Dict
from tqdm import tqdm

import numpy as np

from models.multimodal_classification_model import MultimodalClassificationModel
from datasets.audio_classification_dataset import AudioClassificationDataset

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix

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

if __name__ == '__main__':
    config = add_arguments()
    config = Dict(config)
    
    # this is the root path of the extended dataset
    extended_root_path = config.training.ext_root_path 
    # this is the root path of the model that we want to use for inference
    models_root_path = config.training.ext_model_path 
    subfolders = ["DDK1" , "monologue", "readtext"]
    classes = ["HC", "PD"] # also subsubfolders
    
    audio_paths = []
    labels = []
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    for sf in subfolders:
        for c in classes:
            if sf == "words":
                # find another level of subfolders
                subsubfolders = os.listdir(os.path.join(extended_root_path, sf, c))
                for ssf in subsubfolders:
                    files = os.listdir(os.path.join(extended_root_path, sf, c, ssf))
                    for f in files:
                        audio_paths.append(os.path.join(extended_root_path, sf, c, ssf, f))
                        labels.append(c)
            else:
                files = os.listdir(os.path.join(extended_root_path, sf, c))
                for f in files:
                    audio_paths.append(os.path.join(extended_root_path, sf, c, f))
                    labels.append(c)
                
    print("Number of audio files: ", len(audio_paths))
    print("Number of labels: ", len(labels))
    class_mapping = {"HC": 0, "PD": 1}
    
    config.model.num_classes = len(set(labels))
    model = MultimodalClassificationModel(config=config)
    model = model.eval()
    model = model.to(device)
    
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
        batch_size=8,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    loss_fn = torch.nn.BCELoss()
    
    all_metrics = []
    all_predictions = {}
    all_references = {}
    
    for fold in range(1, 11):
        pt_model_path = os.path.join(models_root_path, f"fold_{fold}.pt")
        model.load_state_dict(torch.load(pt_model_path))
        eval_loss, reference, predictions = eval_one_epoch(model, dl, torch.device("cuda"), loss_fn, is_binary_classification=True)
        all_predictions[fold] = predictions
        all_references[fold] = reference
        m_dict = compute_metrics(reference, predictions, is_binary_classification=True)
        print("=======================================")
        print(f"Fold {fold}")
        for k, v in m_dict.items():
            print(f"{k}: {v*100:.2f}")
        all_metrics.append(m_dict)
        
    print("=======================================")
    print("Average metrics")

    for k in all_metrics[0].keys():
        avg_metric = np.mean([m[k] for m in all_metrics])
        std_metric = np.std([m[k] for m in all_metrics])
        print(f"{k}: {avg_metric*100:.2f} +/- {std_metric*100:.2f}")
        
    print("=======================================")
    
    # majority voting
    final_predictions = []
    for i in range(len(all_predictions[1])):
        votes = [all_predictions[fold][i] for fold in range(1, 11)]
        # all_predictions[fold][i] is an array of 1 element
        votes = [v[0] for v in votes]
        final_predictions.append(int(np.mean(votes) > 0.5))
        
    final_reference = all_references[1]
    m_dict = compute_metrics(final_reference, final_predictions, is_binary_classification=True)
    print("=======================================")
    print("Majority voting metrics")
    for k, v in m_dict.items():
        print(f"{k}: {v*100:.2f}")

