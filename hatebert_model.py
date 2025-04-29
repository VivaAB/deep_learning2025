# -*- coding: utf-8 -*-

"""
This module handles the detection of hate speech using the hateBERT model.

Functions:
    load_hatebert(): Loads the hateBERT tokenizer and model
    f1(): Computes the F1 score
    acc(): Computes the accuracy
    train_epoch(): Trains the model for one epoch
    evaluate(): Evaluates the model on the test set
    plot_training(): Plots the training and test loss and metrics
    update_metrics_log(): Updates the metrics log
    predict_hate_speech(): Predicts the hate speech probability for a given text
    main(): Main function to load the model and test it on a few examples
"""

__author__ = "Marc Bonhôte, Mikaël Schär, Viva ..."
__version__ = "1.0.0"
__status__ = "Development"

# =========== Importing libraries ===========
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm
import evaluate
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

id2label = {0: "Negative", 1: "Positive"}
label2id = {"Negative": 0, "Positive": 1}

# =========== Functions ===========
def load_hatebert():
    # Load the tokenizer and model
    model_name = "GroNLP/hateBERT"
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, id2label=id2label, label2id=label2id)
    return tokenizer, model

def tokenize_function(tokenizer, data, column_name="text"):
    '''
     padding: 'max_length': pad to a length specified by the max_length argument or the
     maximum length accepted by the model if no max_length is provided (max_length=None).
     Padding will still be applied if you only provide a single sequence. [from documentation]


     truncation: True or 'longest_first': truncate to a maximum length specified
     by the max_length argument or the maximum length accepted by the model if
     no max_length is provided (max_length=None). This will truncate the token by
     token, removing a token from the longest sequence in the pair until the
     proper length is reached. [from documentation]
    '''
    return tokenizer(data[column_name], padding="max_length", truncation=True)

def train_epoch_hatebert(model, num_epochs, train_dataloader, optimizer, lr_scheduler, device, num_training_steps):
    progress_bar = tqdm(range(num_training_steps))
    
    # put the model in train mode
    model.train()

    # iterate over epochs
    for epoch in range(num_epochs):
        # iterate over batches in training set
        for batch in train_dataloader:
            # Print the batch type and structure to debug
            print("Batch type:", type(batch))
            
            # Handle the case where batch is a dictionary
            if isinstance(batch, dict):
                # Create a new dictionary with only the keys we need
                model_inputs = {}
                
                # Check if each key exists and if the value is a tensor
                if "input_ids" in batch and hasattr(batch["input_ids"], "to"):
                    model_inputs["input_ids"] = batch["input_ids"].to(device)
                
                if "attention_mask" in batch and hasattr(batch["attention_mask"], "to"):
                    model_inputs["attention_mask"] = batch["attention_mask"].to(device)
                
                if "labels" in batch and hasattr(batch["labels"], "to"):
                    model_inputs["labels"] = batch["labels"].to(device)
                
                # Forward pass
                outputs = model(**model_inputs)
            else:
                # If batch is not a dictionary, handle it as a list
                # Assuming the list contains tensors in the order: input_ids, attention_mask, labels
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device)
                labels = batch[2].to(device)
                
                # Create a dictionary with the expected keys
                model_inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels
                }
                
                # Forward pass
                outputs = model(**model_inputs)
            
            loss = outputs.loss

            # do the backward pass
            loss.backward()

            # perform one step of the optimizer
            optimizer.step()

            # peform one step of the lr_scheduler, similar with the optimizer
            lr_scheduler.step()

            # zero the gradients, call zero_grad() on the optimizer
            optimizer.zero_grad()

            progress_bar.update(1)

def evaluate_hatebert(model, test_dataloader, device):
    # define the metric you want to use to evaluate your model
    metric = evaluate.load("accuracy")
    progress_bar = tqdm(range(len(test_dataloader)))

    # put the model in eval mode
    model.eval()
    # iterate over batches of evaluation dataset
    for batch in test_dataloader:
        # Print the batch type and structure to debug
        print("Batch type:", type(batch))
        
        # Handle the case where batch is a dictionary
        if isinstance(batch, dict):
            # Create a new dictionary with only the keys we need
            model_inputs = {}
            
            # Check if each key exists and if the value is a tensor
            if "input_ids" in batch and hasattr(batch["input_ids"], "to"):
                model_inputs["input_ids"] = batch["input_ids"].to(device)
            
            if "attention_mask" in batch and hasattr(batch["attention_mask"], "to"):
                model_inputs["attention_mask"] = batch["attention_mask"].to(device)
            
            if "labels" in batch and hasattr(batch["labels"], "to"):
                model_inputs["labels"] = batch["labels"].to(device)
            
            # Forward pass
            outputs = model(**model_inputs)
        else:
            # If batch is not a dictionary, handle it as a list
            # Assuming the list contains tensors in the order: input_ids, attention_mask, labels
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)
            
            # Create a dictionary with the expected keys
            model_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }
            
            # Forward pass
            outputs = model(**model_inputs)


        logits = outputs.logits

        # use argmax to get the predicted class
        predictions = torch.argmax(logits, dim=-1)
        
        metric.add_batch(predictions=predictions, references=batch["labels"])
        progress_bar.update(1)
    # calculate a metric by  calling metric.compute()
    print(metric.compute())

def evaluate_hatebert_with_bias(model, dataloader, device):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    all_target_groups = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Print the batch type and structure to debug
            print("Batch type:", type(batch))
            
            # Handle the case where batch is a dictionary
            if isinstance(batch, dict):
                # Create a new dictionary with only the keys we need
                model_inputs = {}
                
                # Check if each key exists and if the value is a tensor
                if "input_ids" in batch and hasattr(batch["input_ids"], "to"):
                    model_inputs["input_ids"] = batch["input_ids"].to(device)
                
                if "attention_mask" in batch and hasattr(batch["attention_mask"], "to"):
                    model_inputs["attention_mask"] = batch["attention_mask"].to(device)
                
                if "labels" in batch and hasattr(batch["labels"], "to"):
                    model_inputs["labels"] = batch["labels"].to(device)
                
                # Forward pass
                outputs = model(**model_inputs)
            else:
                # If batch is not a dictionary, handle it as a list
                # Assuming the list contains tensors in the order: input_ids, attention_mask, labels
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device)
                labels = batch[2].to(device)
                
                # Create a dictionary with the expected keys
                model_inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels
                }
                
                # Forward pass
                outputs = model(**model_inputs)
            
            # Get predictions and labels
            predictions = torch.argmax(outputs.logits, dim=-1)
            labels = batch["labels"]
            target_groups = batch["target_group"]
            
            # Store results
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_target_groups.extend(target_groups)
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_target_groups = np.array(all_target_groups)
    
    # Calculate overall accuracy
    overall_accuracy = accuracy_score(all_labels, all_predictions)
    print(f"Overall Accuracy: {overall_accuracy:.4f}")
    
    # Calculate accuracy for each target group
    unique_groups = np.unique(all_target_groups)
    for group in unique_groups:
        group_mask = all_target_groups == group
        group_accuracy = accuracy_score(all_labels[group_mask], all_predictions[group_mask])
        print(f"Accuracy for target group '{group}': {group_accuracy:.4f}")
        print(f"Number of samples in group '{group}': {np.sum(group_mask)}")
        
        # Calculate confusion matrix for this group
        cm = confusion_matrix(all_labels[group_mask], all_predictions[group_mask])
        print(f"Confusion Matrix for '{group}':")
        print(cm)
        print()
    
    return overall_accuracy

def main():
    print("Running hatebert_dectection.py Library")

if __name__ == "__main__":
    main() 