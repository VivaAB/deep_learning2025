# -*- coding: utf-8 -*-

# Portions of this code were written or refined with assistance from Cursor AI (https://www.cursor.sh)

"""
ToxiGen Dataset Handler

This module provides a comprehensive interface for loading, processing, and analyzing the ToxiGen dataset.
The ToxiGen dataset is a large-scale dataset of toxic and non-toxic language examples, with human annotations
and standardized target groups. This module handles data loading, preprocessing, label standardization,
and provides statistical analysis capabilities.

Key Features:
- Dataset loading from HuggingFace
- Target group standardization
- Label processing and annotation
- Statistical analysis and visualization
- Data cleaning and preprocessing

Classes:
    ToxiGenDataset: Main class for handling all ToxiGen dataset operations
"""

__author__ = "Marc Bonhôte, Mikaël Schär, Viva Berlenghi"
__status__ = "Final"

import os
import pandas as pd
from datasets import load_dataset, Dataset
from toxigen import label_annotations
from typing import Dict, Optional
import matplotlib.pyplot as plt

class ToxiGenDataset:
    def __init__(self, huggingface_token: Optional[str] = None):
        """
        Initialize the ToxiGenDataset class.

        This constructor sets up the dataset handler with necessary configurations including:
        - HuggingFace authentication
        - Target group standardization mapping
        - Column removal specifications

        Args:
            huggingface_token (Optional[str]): HuggingFace API token for dataset access.
                                              Required if dataset is not publicly accessible.
        """
        self.huggingface_token = huggingface_token
        if huggingface_token:
            os.environ["HUGGING_FACE_HUB_TOKEN"] = huggingface_token
        
        # Mapping for standardizing target group names
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
        
        # Columns to be removed during preprocessing
        self.columns_to_remove = [
            'factual?', 'ingroup_effect', 'lewd', 'framing',
            'predicted_group', 'stereotyping', 'intent',
            'toxicity_ai', 'toxicity_human', 'predicted_author',
            'actual_method'
        ]
        
        self.dataset = None

    def load_dataset(self) -> Dataset:
        """
        Load and preprocess the ToxiGen dataset.

        This method:
        1. Loads the dataset from HuggingFace
        2. Processes labels for both train and test splits
        3. Standardizes target groups
        4. Removes unnecessary columns

        Returns:
            Dataset: Processed HuggingFace dataset with standardized labels and target groups
        """
        # Load the annotated dataset from HuggingFace
        self.dataset = load_dataset("toxigen/toxigen-data", name="annotated")
        
        # Process labels for both splits
        for split in ['train', 'test']:
            self._process_labels(split)
        
        # Post-process and standardize target groups
        self._post_process()
        
        return self.dataset

    def _process_labels(self, split: str):
        """
        Process and annotate labels for a specific dataset split.

        This method:
        1. Converts the split to a pandas DataFrame
        2. Applies the label_annotations function
        3. Adds processed labels back to the dataset

        Args:
            split (str): Dataset split to process ('train' or 'test')
        """
        # Convert to pandas DataFrame for label processing
        df = pd.DataFrame(self.dataset[split])
        
        # Get processed labels using the label_annotations function
        labels = label_annotations(df)
        
        # Add processed labels to the dataset
        new_examples = []
        for example, label in zip(self.dataset[split], labels['label']):
            new_example = example.copy()
            new_example['labels'] = label
            new_examples.append(new_example)
        
        # Update the split with the new labeled data
        self.dataset[split] = Dataset.from_list(new_examples)

    def _post_process(self):
        """
        Post-process the dataset by standardizing target groups and removing unnecessary columns.

        This method:
        1. Removes specified columns from both splits
        2. Standardizes target group names using the standardization map
        3. Updates the dataset with processed data
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
            
            # Update the split with standardized data
            self.dataset[split] = Dataset.from_list(new_examples)

    def get_dataset(self) -> Optional[Dataset]:
        """
        Retrieve the processed dataset.

        Returns:
            Optional[Dataset]: The processed dataset if loaded, None otherwise
        """
        return self.dataset

    def print_statistics(self):
        """
        Generate and display comprehensive statistics about the dataset.

        This method provides:
        1. A visualization of hate vs non-hate distribution across target groups
        2. Number of examples per split
        3. Label distribution (hate vs non-hate) for each split
        4. Target group distribution statistics

        The visualization is saved as 'ToxiGen_train_dataset_distribution.png'
        """
        if self.dataset is None:
            print("Dataset not loaded. Call load_dataset() first.")
            return

        # Get data from train split
        train_groups = self.dataset['train']['target_group']
        train_labels = self.dataset['train']['labels']

        # Calculate total counts per group for sorting
        total_counts = {}
        for group in set(train_groups):
            hate_count = sum(1 for g, l in zip(train_groups, train_labels) if g == group and l == 1)
            nonhate_count = sum(1 for g, l in zip(train_groups, train_labels) if g == group and l == 0)
            total_counts[group] = hate_count + nonhate_count
            
        # Sort groups by total count in descending order
        unique_groups = sorted(set(train_groups), key=lambda x: total_counts[x], reverse=True)

        # Calculate counts for each group
        group_hate_counts = {}
        group_nonhate_counts = {}
        
        for group, label in zip(train_groups, train_labels):
            if label == 1:  # hate speech
                group_hate_counts[group] = group_hate_counts.get(group, 0) + 1
            else:  # non-hate speech
                group_nonhate_counts[group] = group_nonhate_counts.get(group, 0) + 1

        # Create lists for plotting
        hate_counts = [group_hate_counts.get(group, 0) for group in unique_groups]
        nonhate_counts = [group_nonhate_counts.get(group, 0) for group in unique_groups]

        # Create the visualization
        plt.figure(figsize=(15, 8))
        
        # Create stacked bars
        plt.bar(unique_groups, hate_counts, color='#ff0000', label='Hate')
        plt.bar(unique_groups, nonhate_counts, bottom=hate_counts, color='#808080', label='Non-hate')

        # Customize the plot
        plt.xticks(rotation=45, ha='right')
        plt.xlabel('Target Group')
        plt.ylabel('Number of data points')
        plt.legend()

        # Add value labels in the middle of each segment
        for i, (hate, nonhate) in enumerate(zip(hate_counts, nonhate_counts)):
            # Label for hate segment
            plt.text(i, hate/2, f'{hate:,}', ha='center', va='center', color='white')
            # Label for non-hate segment
            plt.text(i, hate + nonhate/2, f'{nonhate:,}', ha='center', va='center', color='white')

        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Save and display the plot
        plt.savefig('ToxiGen_train_dataset_distribution.png', dpi=300)
        plt.show()

        # Print detailed statistics
        print("=== Dataset Statistics ===\n")
        
        # Print number of examples per split
        print("Number of examples:")
        for split in ['train', 'test']:
            print(f"- {split}: {len(self.dataset[split])}")
        print()

        # Print label distribution
        print("Label Distribution:")
        for split in ['train', 'test']:
            labels = self.dataset[split]['labels']
            total = len(labels)
            toxic = sum(labels)
            non_toxic = total - toxic
            print(f"\n{split.capitalize()} split:")
            print(f"- Hate speech: {toxic} ({toxic/total*100:.1f}%)")
            print(f"- Non-hate speech: {non_toxic} ({non_toxic/total*100:.1f}%)")
        print()

        # Print target group distribution
        print("Target Group Distribution:")
        for split in ['train', 'test']:
            target_groups = self.dataset[split]['target_group']
            group_counts = {}
            for group in target_groups:
                group_counts[group] = group_counts.get(group, 0) + 1
            
            print(f"\n{split.capitalize()} split:")
            for group, count in sorted(group_counts.items()):
                print(f"- {group}: {count} ({count/len(target_groups)*100:.1f}%)")
        print()


