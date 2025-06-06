import torch
from sklearn.metrics import f1_score, accuracy_score

from torch.nn.utils.rnn import pad_sequence

from datasets import Dataset
from typing import Dict
from collections import defaultdict

"""Below are some utils functions that are used to plot the loss curves and update the loss log."""
def update_metrics_log(metrics_names, metrics_log, new_metrics_dict):
    '''
    - metrics_names: the keys/names of the logged metrics
    - metrics_log: existing metrics log that will be updated
    - new_metrics_dict: epoch_metrics output from train_epoch and evaluate functions
    '''
    for i in range(len(metrics_names)):
        curr_metric_name = metrics_names[i]
        metrics_log[i].append(new_metrics_dict[curr_metric_name])
    return metrics_log

"""## Metrics

##### 3.2 Introducing simple metrics, such as accuracy and F1-score.
"""

def f1(preds, target):
    return f1_score(target, preds, average='macro')

def acc(preds, target):
    return accuracy_score(target, preds)

"""## Data collator"""

def custom_collate_fn(batch):
    input_ids = [torch.tensor(example["input_ids"]) for example in batch]
    attention_mask = [torch.tensor(example["attention_mask"]) for example in batch]
    labels = [torch.tensor(example["labels"]) for example in batch]
    target_groups = [example["target_group"] for example in batch] # keep as list of strings
    token_type_ids = [torch.tensor(example.get("token_type_ids", [])) for example in batch]  # optional, if your model uses it

    return {
        "input_ids": pad_sequence(input_ids, batch_first=True, padding_value=0),
        "attention_mask": pad_sequence(attention_mask, batch_first=True, padding_value=0),
        "labels": torch.tensor(labels),
        "target_group": target_groups,
        "token_type_ids": pad_sequence(token_type_ids, batch_first=True, padding_value=0) if token_type_ids else None
    }

def remove_group(dataset) -> Dict[str, Dataset]:
        """
        Removes examples from the dataset where the target group appears less than the threshold.

        Args:
            dataset: Dictionary containing the dataset splits.
            threshold: Minimum number of occurrences for a target group to be kept.

        Returns:
            Dictionary containing the dataset splits with small target groups removed.
        """

        threshold = {'train': 100, 'test': 20} # group above are keep
        print(f"Removing target groups with fewer than {threshold} occurrences for training data...")
        new_dataset = {}

        for split in ['train', 'test']:
            print(f"Processing {split} split...")
            examples = dataset[split]

            # Count occurrences of each target group
            target_group_counts = defaultdict(int)
            for example in examples:
                # Assuming 'target_group' is a list, take the first element
                group = example['target_group'][0] if example['target_group'] else 'none'
                target_group_counts[group] += 1

            # Identify target groups to keep
            target_groups_to_keep = {group for group, count in target_group_counts.items() if count >= threshold[split]}

            # Filter examples
            filtered_examples = []
            for example in examples:
                group = example['target_group'][0] if example['target_group'] else 'none'
                if group in target_groups_to_keep:
                    if len(example['target_group']) > 1:
                        example['target_group'] = [example['target_group'][0]]
                    filtered_examples.append(example)

            # Create a new Dataset object from the filtered list
            new_dataset[split] = Dataset.from_list(filtered_examples)
            print(f"Removed {len(examples) - len(filtered_examples)} examples from {split} split.")
            print(f"Remaining examples in {split} split: {len(filtered_examples)}")

        return new_dataset

"""Training for one epoch is unlikely to be helpful, which is why we usually train the model for a number of epochs.

**Note**: you have to use validation for each step of training, but now we will focus only on the toy example and will track the performance on test set.
"""
def train_cycle(model, optimizer, lr_scheduler, train_loader, test_loader, n_epochs, num_training_steps=None):

    for epoch in range(n_epochs):
        print("Epoch {0} of {1}".format(epoch, n_epochs - 1))
        model.train_epoch_BERT(
            train_loader,
            optimizer,
            lr_scheduler,
            num_training_steps
        )

        model.evaluate_BERT_with_bias(test_loader)