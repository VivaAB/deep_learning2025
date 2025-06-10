import torch
from sklearn.metrics import f1_score, accuracy_score

from torch.nn.utils.rnn import pad_sequence

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

"""
Training for one epoch is unlikely to be helpful, which is why we usually train the model for a number of epochs.
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