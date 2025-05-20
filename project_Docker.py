from loading_dataset_toxigen import load_toxigen, post_process_toxigen
from hatebert_model import load_hatebert, tokenize_function, train_epoch_hatebert, evaluate_hatebert, evaluate_hatebert_with_bias
from torch.utils.data import DataLoader
import torch
import torch.nn as  nn
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
import argparse

import os
from pathlib import Path
import matplotlib.pyplot as plt
from checkpoint_utils import save_checkpoint, load_checkpoint


def parse_args():
    #when working with python files from console it's better to specify
    parser = argparse.ArgumentParser(description="File creation script.")
    #parser.add_argument("--dataset_path", required=True, help="Dataset path")
    parser.add_argument("--results_path", required=True, help="Output directory")

    args = parser.parse_args()

    return args.results_path


RESULTS_DIR = parse_args()
if not os.path.exists(RESULTS_DIR):
    os.mkdir(RESULTS_DIR)
#directory to store model checkpoints to
if not os.path.exists(Path(RESULTS_DIR) / 'checkpoints'):
    os.mkdir(Path(RESULTS_DIR) / 'checkpoints')


toxigen = load_toxigen()

print(toxigen)


tokenizer, model_hatebert = load_hatebert()


'''
 apply the function to all the elements in the dataset (individually or in batches)
 https://huggingface.co/docs/datasets/v1.11.0/package_reference/main_classes.html?highlight=dataset%20map#datasets.Dataset.map
 batch mode is very powerful. It allows you to speed up processing
 more info here: https://huggingface.co/docs/datasets/en/about_map_batch
'''
cache_files = {
    "test": ".cache/datasets/toxigen/toxigen_test_tokenized.arrow",
    "train": ".cache/datasets/toxigen/toxigen_train_tokenized.arrow"
} #path to the local cache files, where the current computation from the following function will be stored. 
# Caching saves RAM when working with large datasets and saves time instead of doing transformations on the fly.
tokenized_toxigen = toxigen.map(lambda x: tokenize_function(tokenizer, x, "text"), batched=True, cache_file_names=cache_files)

print(tokenized_toxigen)


tokenized_toxigen = post_process_toxigen(tokenized_toxigen)

# create a smaller subset of the dataset as previously shown to speed up the fine-tuning

small_train_dataset = tokenized_toxigen["train"].shuffle(seed=42).select(range(50))
small_eval_dataset = tokenized_toxigen["test"].shuffle(seed=42).select(range(50))

train_dataset = tokenized_toxigen["train"]
eval_dataset = tokenized_toxigen["test"]

# create a DataLoader for your training and test datasets so you can iterate over batches of data:
train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=8)
test_dataloader = DataLoader(small_eval_dataset, batch_size=8)

print("HateBERT number of parameters: ", model_hatebert.num_parameters())


"""## Metrics

##### 3.2 Introducing simple metrics, such as accuracy and F1-score.
"""

def f1(preds, target):
    return f1_score(target, preds, average='macro')

def acc(preds, target):
    return accuracy_score(target, preds)


"""Below are some utils functions that are used to plot the loss curves and update the loss log."""

def plot_training(train_loss, test_loss, metrics_names, train_metrics_logs, test_metrics_logs):
    fig, ax = plt.subplots(1, len(metrics_names) + 1, figsize=((len(metrics_names) + 1) * 5, 5))

    ax[0].plot(train_loss, c='blue', label='train')
    ax[0].plot(test_loss, c='orange', label='test')
    ax[0].set_title('Loss')
    ax[0].set_xlabel('epoch')
    ax[0].legend()

    for i in range(len(metrics_names)):
        ax[i + 1].plot(train_metrics_logs[i], c='blue', label='train')
        ax[i + 1].plot(test_metrics_logs[i], c='orange', label='test')
        ax[i + 1].set_title(metrics_names[i])
        ax[i + 1].set_xlabel('epoch')
        ax[i + 1].legend()

    plt.savefig(Path(RESULTS_DIR) / "training loss and metrics.jpg")

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


"""Training for one epoch is unlikely to be helpful, which is why we usually train the model for a number of epochs.

**Note**: you have to use validation for each step of training, but now we will focus only on the toy example and will track the performance on test set.
"""
def train_cycle(model, optimizer, criterion, metrics, train_loader, test_loader, n_epochs, device,
                store_checkpoint_for_every_epoch, start_epoch=0):
    train_loss_log,  test_loss_log = [], []
    metrics_names = list(metrics.keys())
    train_metrics_log = [[] for i in range(len(metrics))]
    test_metrics_log = [[] for i in range(len(metrics))]


    for epoch in range(start_epoch, n_epochs):
        print("Epoch {0} of {1}".format(epoch, n_epochs - 1))
        train_loss, train_metrics = train_epoch_hatebert(model_hatebert, num_epochs, train_dataloader, optimizer, lr_scheduler, criterion, metrics, device, num_training_steps)

        overall_acc, test_loss, test_metrics = evaluate_hatebert_with_bias(model, criterion, metrics, test_loader, device)

        train_loss_log.append(train_loss)
        train_metrics_log = update_metrics_log(metrics_names, train_metrics_log, train_metrics)

        test_loss_log.append(test_loss)
        test_metrics_log = update_metrics_log(metrics_names, test_metrics_log, test_metrics)

        plot_training(train_loss_log, test_loss_log, metrics_names, train_metrics_log, test_metrics_log)

        save_checkpoint(model, optimizer, epoch, loss=train_loss, checkpoint_path = Path(RESULTS_DIR) / "checkpoints/checkpoint.pth",
                        store_checkpoint_for_every_epoch=store_checkpoint_for_every_epoch)
    return train_metrics_log, test_metrics_log

#optimizer
from torch.optim import AdamW

optimizer = AdamW(model_hatebert.parameters(), lr=5e-5)

#Scheduler ???
from transformers import get_scheduler

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
# feel free to experiment with different num_warmup_steps
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=1, num_training_steps=num_training_steps
)


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model_hatebert.to(device)

'''
train_epoch_hatebert(model_hatebert, num_epochs, train_dataloader, optimizer, lr_scheduler, device, num_training_steps)

evaluate_hatebert(model_hatebert, test_dataloader, device)

evaluate_hatebert_with_bias(model_hatebert, test_dataloader, device)
'''


saved_epoch, _ = load_checkpoint(model_hatebert, optimizer, checkpoint_path = Path(RESULTS_DIR) / "checkpoints/checkpoint.pth")
if saved_epoch == 0:
    start_epoch = 0
else:
    start_epoch = saved_epoch + 1  #if the checkpoint from the epoch saved_epoch is stored, we want to start the training from the next epoch

criterion = nn.CrossEntropyLoss()

metrics = {'ACC': acc, 'F1-weighted': f1}
criterion.to(device)


train_metrics_log, test_metrics_log = train_cycle(model_hatebert, optimizer, criterion, metrics, train_dataloader, test_dataloader,
                                                  n_epochs=num_epochs, device=device, store_checkpoint_for_every_epoch=False,
                                                  start_epoch=start_epoch)