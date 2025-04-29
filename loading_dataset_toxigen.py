# -*- coding: utf-8 -*-

"""
This module handles loading and preprocessing of the ToxiGen dataset.

The ToxiGen dataset contains examples of toxic and non-toxic language,
with human annotations. This module provides functions to load the dataset
and process the annotations into labels.

Functions:
    load_toxigen(): Loads and preprocesses the ToxiGen dataset
    
Returns:
    Tuple of (train_set, test_set) as pandas DataFrames with labels
"""

__author__ = "Marc Bonhôte, Mikaël Schär, Viva ..."
__version__ = "1.0.0"
__status__ = "Development"


import pandas as pd
from toxigen import label_annotations
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
import torch
import os

# To complete with your own token
os.environ["HUGGING_FACE_HUB_TOKEN"] = ""

def load_toxigen():
    # Load the dataset
    #train_data = load_dataset("toxigen/toxigen-data", name="train") # 250k training examples
    annotated_data = load_dataset("toxigen/toxigen-data", name="annotated") # Human study
    #raw_annotations = load_dataset("toxigen/toxigen-data", name="annotations") # Raw Human study
   
    # Convert both train and test splits to pandas DataFrames
    train_df = pd.DataFrame(annotated_data['train'])
    test_df = pd.DataFrame(annotated_data['test'])
    
    # Apply label_annotations function to get binary labels for both splits
    train_labels = label_annotations(train_df)
    test_labels = label_annotations(test_df)
    
    # Add labels to the train and test sets
    train_df['labels'] = train_labels['label']
    test_df['labels'] = test_labels['label']
    
    # Convert back to HuggingFace dataset format
    annotated_data['train'] = Dataset.from_pandas(train_df)
    annotated_data['test'] = Dataset.from_pandas(test_df)
    
    return annotated_data

def post_process_toxigen(tokenized_dataset):
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])
    tokenized_dataset.set_format("torch")
    return tokenized_dataset


