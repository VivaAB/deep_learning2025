# -*- coding: utf-8 -*-

"""
This module handles loading and preprocessing of the ToxiGen dataset.

The ToxiGen dataset contains examples of toxic and non-toxic language,
with human annotations. This module provides a ToxiGenDataset class to load
and process the dataset with standardized target groups and processed labels.

Classes:
    ToxiGenDataset: Main class for handling ToxiGen dataset operations
"""

__author__ = "Marc Bonhôte, Mikaël Schär, Viva ?"
__status__ = "Development"

import os
import pandas as pd
from datasets import load_dataset, Dataset
from toxigen import label_annotations
from typing import Dict, Optional

class ToxiGenDataset:
    def __init__(self, huggingface_token: Optional[str] = None):
        """
        Initialize the ToxiGenDataset class.

        Args:
            huggingface_token: Optional HuggingFace token for dataset access
        """
        self.huggingface_token = huggingface_token
        if huggingface_token:
            os.environ["HUGGING_FACE_HUB_TOKEN"] = huggingface_token
        
        # Define standardization map for target groups
        self.standardization_map = {
            'asian folks': 'asian',
            'black folks / african-americans': 'black',
            'black/african-american folks': 'black',
            'chinese folks': 'chinese',
            'folks with mental disabilities': 'mental_dis',
            'folks with physical disabilities': 'physical_dis',
            'jewish folks': 'jewish',
            'latino/hispanic folks': 'latino',
            'lgbtq+ folks': 'lgbtq',
            'mexican folks': 'mexican',
            'middle eastern folks': 'middle_east',
            'muslim folks': 'muslim',
            'native american folks': 'native_american',
            'native american/indigenous folks': 'native_american',
            'women': 'women'
        }
        
        # Define columns to remove
        self.columns_to_remove = [
            'factual?', 'ingroup_effect', 'lewd', 'framing',
            'predicted_group', 'stereotyping', 'intent',
            'toxicity_ai', 'toxicity_human', 'predicted_author',
            'actual_method'
        ]
        
        self.dataset = None

    def load_dataset(self) -> Dataset:
        """
        Load the ToxiGen dataset, standardize target groups, and process labels.

        Returns:
            HuggingFace dataset with processed labels and standardized target groups
        """
        # Load the annotated dataset
        self.dataset = load_dataset("toxigen/toxigen-data", name="annotated")
        
        # Process labels for both splits
        for split in ['train', 'test']:
            self._process_labels(split)
        
        # Post-process and standardize target groups
        self._post_process()
        
        return self.dataset

    def _process_labels(self, split: str):
        """
        Process labels for a specific dataset split.

        Args:
            split: Dataset split to process ('train' or 'test')
        """
        # Convert to pandas DataFrame for label_annotations
        df = pd.DataFrame(self.dataset[split])
        
        # Get labels using the label_annotations function
        labels = label_annotations(df)
        
        # Add labels to the dataset
        new_examples = []
        for example, label in zip(self.dataset[split], labels['label']):
            new_example = example.copy()
            new_example['labels'] = label
            new_examples.append(new_example)
        
        # Replace the split with the new data including labels
        self.dataset[split] = Dataset.from_list(new_examples)

    def _post_process(self):
        """
        Post-process the dataset by removing unnecessary columns and standardizing target groups.
        """
        for split in ['train', 'test']:
            # Remove unnecessary columns
            self.dataset[split] = self.dataset[split].remove_columns(self.columns_to_remove)
            
            # Standardize target groups
            new_examples = []
            for example in self.dataset[split]:
                new_example = example.copy()
                new_example['target_group'] = self.standardization_map.get(
                    example['target_group'],
                    example['target_group']
                )
                new_examples.append(new_example)
            
            # Replace the split with the new standardized data
            self.dataset[split] = Dataset.from_list(new_examples)

    def get_dataset(self) -> Optional[Dataset]:
        """
        Get the processed dataset.

        Returns:
            The processed dataset or None if not loaded
        """
        return self.dataset


