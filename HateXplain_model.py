from torch.utils.data import DataLoader
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import confusion_matrix

from tqdm import tqdm
import numpy as np
from collections import defaultdict

from typing import Optional, Callable

import numpy as np
from tqdm import tqdm

"""Model training and evaluation"""
class HateXplainModel:
    def __init__(
        self,
        model_name,
        num_labels: int = 3,
        device: Optional[str] = None,
        max_length: int = 512,
        f1: Optional[Callable] = None,
        acc: Optional[Callable] = None
    ):
        """
        Initialize the HateXplain classifier.

        Args:
            model_name: Name of the pre-trained model to use
            num_labels: Number of classification labels (default: 3 for multi-classes classification)
            device: Device to run the model on ('cuda' or 'cpu')
            max_length: Maximum sequence length for tokenization (default: 512)
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.device = device
        self.max_length = max_length

        # Set device
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Initialize model and tokenizer
        self.tokenizer = None
        self.model = None
        self.load_sc_weighted_BERT()

        # Training state
        self.optimizer = None
        self.scheduler = None
        self.metrics = {
            'ACC': acc,
            'F1-weighted': f1
        }

    def load_sc_weighted_BERT(self):
        """
        Load the pre-trained SC-weighted BERT model and tokenizer.

        Returns:
            model: Pre-trained SC-weighted BERT model
            tokenizer: Tokenizer for the SC-weighted BERT model
        """
        # Load the pre-trained model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels = self.num_labels
        )

        self.model.to(self.device)

    def tokenize_function(self, data, column_name="text"):
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
        return self.tokenizer(data[column_name], padding="max_length", truncation=True)

    def train_epoch_BERT(
        self,
        train_dataloader: DataLoader,
        optimizer,
        lr_scheduler,
        num_training_steps: int = None,
    ):

        progress_bar = tqdm(range(num_training_steps))

        # put the model in train mode
        self.model.train()

        # initialize epoch loss and metrics
        epoch_loss = 0
        epoch_metrics = dict(zip(self.metrics.keys(), torch.zeros(len(self.metrics))))

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
                    model_inputs["input_ids"] = batch["input_ids"].to(self.device)

                if "attention_mask" in batch and hasattr(batch["attention_mask"], "to"):
                    model_inputs["attention_mask"] = batch["attention_mask"].to(self.device)

                if "labels" in batch and hasattr(batch["labels"], "to"):
                    model_inputs["labels"] = batch["labels"].to(self.device)

                # Forward pass
                outputs = self.model(**model_inputs)
            else:
                # If batch is not a dictionary, handle it as a list
                # Assuming the list contains tensors in the order: labels, input_ids, attention_mask
                labels = batch['labels'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                # Create a dictionary with the expected keys
                model_inputs = {
                    "labels": labels,
                    "input_ids": input_ids,
                    "attention_mask": attention_mask
                }

                # Forward pass
                outputs = self.model(**model_inputs)

            # zero the gradients, call zero_grad() on the optimizer
            optimizer.zero_grad()

            loss = outputs.loss

            # do the backward pass
            loss.backward()

            # perform one step of the optimizer
            optimizer.step()

            # peform one step of the lr_scheduler, similar with the optimizer
            lr_scheduler.step()

            '''
            Compute predictions
            Store predictions in variable preds
            No gradients should be propagated at this step
            '''
            with torch.no_grad():
                _, preds = torch.max(outputs.logits, 1)

            '''
            Compute metrics
            Note: no gradients should be propagated at this step
            '''
            with torch.no_grad():
                for k in epoch_metrics.keys():
                    epoch_metrics[k] += self.metrics[k](preds.cpu(), model_inputs["labels"].cpu())

            # log loss statistics
            epoch_loss += loss.item()

            progress_bar.update(1)

        # for the epoch loss, we take the average of the losses computed over the mini-batches
        epoch_loss /= len(train_dataloader)

        # for the epoch loss, we compute the average of the metrics over the mini-batches
        for k in epoch_metrics.keys():
              epoch_metrics[k] /= len(train_dataloader)

        print('train Loss: {:.4f}, '.format(epoch_loss),
              ', '.join(['{}: {:.4f}'.format(k, epoch_metrics[k]) for k in epoch_metrics.keys()]))

    def evaluate_BERT_with_bias(
        self,
        eval_dataloader: DataLoader
    ):

        self.model.eval()
        all_predictions = []
        all_labels = []
        all_target_groups = []

        # initialize epoch loss and metrics
        epoch_loss = 0
        epoch_metrics = dict(zip(self.metrics.keys(), torch.zeros(len(self.metrics))))

        with torch.no_grad():
            for batch in eval_dataloader:
                # Print the batch type and structure to debug

                # Handle the case where batch is a dictionary
                if isinstance(batch, dict):
                    # Create a new dictionary with only the keys we need
                    model_inputs = {}

                    # Check if each key exists and if the value is a tensor
                    if "input_ids" in batch and hasattr(batch["input_ids"], "to"):
                        model_inputs["input_ids"] = batch["input_ids"].to(self.device)

                    if "attention_mask" in batch and hasattr(batch["attention_mask"], "to"):
                        model_inputs["attention_mask"] = batch["attention_mask"].to(self.device)

                    if "labels" in batch and hasattr(batch["labels"], "to"):
                        model_inputs["labels"] = batch["labels"].to(self.device)

                    # Forward pass
                    outputs = self.model(**model_inputs)
                else:
                    # If batch is not a dictionary, handle it as a list
                    # Assuming the list contains tensors in the order: labels, input_ids, attention_mask
                    labels = batch['labels'].to(self.device)
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)

                    # Create a dictionary with the expected keys
                    model_inputs = {
                        "labels": labels,
                        "input_ids": input_ids,
                        "attention_mask": attention_mask
                    }

                    # Forward pass
                    outputs = self.model(**model_inputs)

                # Get predictions and labels
                predictions = torch.argmax(outputs.logits, dim=1)
                labels = batch["labels"]

                target_groups = batch["target_group"]

                loss = outputs.loss

                for k in epoch_metrics.keys():
                    epoch_metrics[k] += self.metrics[k](predictions.cpu().numpy(), model_inputs["labels"].cpu().numpy())

                epoch_loss += loss.item()

                # Store results
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_target_groups.extend(target_groups)

            epoch_loss /= len(eval_dataloader)

            for k in epoch_metrics.keys():
              epoch_metrics[k] /= len(eval_dataloader)

        print('eval Loss: {:.4f}, '.format(epoch_loss),
              ', '.join(['{}: {:.4f}'.format(k, epoch_metrics[k]) for k in epoch_metrics.keys()]))

        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)

        # Calculate overall accuracy
        overall_accuracy = self.metrics['ACC'](all_labels, all_predictions)
        overall_F1 = self.metrics['F1-weighted'](all_labels, all_predictions)
        print(f"Overall Accuracy: {overall_accuracy:.4f}")
        print(f"Overall F1 Score: {overall_F1:.4f}")


        # Dictionnaire pour stocker les prédictions par groupe
        groupwise_preds = defaultdict(list)
        groupwise_labels = defaultdict(list)

        for preds, labels, groups in zip(all_predictions, all_labels, all_target_groups):
            for group in groups:
                groupwise_preds[group].append(preds)
                groupwise_labels[group].append(labels)

        # Évaluation par groupe
        for group in groupwise_preds:
            preds = np.array(groupwise_preds[group])
            labels = np.array(groupwise_labels[group])
            acc = self.metrics['ACC'](labels, preds)
            f1_score = self.metrics['F1-weighted'](labels, preds)
            cm = confusion_matrix(labels, preds)
            print(f"Accuracy for target group '{group}': {acc:.4f}")
            print(f"F1 Score for target group '{group}': {f1_score:.4f}")
            print(f"Confusion Matrix for '{group}':")
            print(cm)
            print()