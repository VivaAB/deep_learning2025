from collections import defaultdict

from datasets import load_dataset, Dataset
from typing import Dict, Optional

"""Dataset"""
class HateXplainDataset:
    def __init__(self):
        """
        Initialize the HateXplainDataset class.
        """
        self.dataset = None
        self._processed_dataset = None

    def load_dataset(self) -> Dict[str, Dataset]:
        """
        Load the HateXplain dataset from HuggingFace.

        Returns:
            Dictionary containing the dataset splits
        """
        try:
            print("Attempting to load HateXplain dataset...")
            self.dataset = load_dataset("hatexplain", trust_remote_code=True)
            # After loading the dataset
            for split in ['train', 'validation', 'test']:
                self.dataset[split] = self._process_labels(split)

            # Post-process and standardize target groups
            #self._post_process()

            print("Dataset loaded successfully")
            print(f"Available splits: {list(self.dataset.keys())}")

            return {
                'train': self.dataset['train'],
                'test': self.dataset['test']
            }

        except FileNotFoundError as e:
            print(f"Dataset not found: {str(e)}")
            raise
        except ValueError as e:
            print(f"Value error: {str(e)}")
            raise
        except FileNotFoundError as e:
            print(f"Dataset not found: {str(e)}")
            raise
        except ValueError as e:
            print(f"Value error: {str(e)}")
            raise


        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            raise

    def _process_labels(self, split: str):
        """
        Process labels for a specific dataset split.

        Args:
            split: Dataset split to process ('train', 'validation', or 'test')
        """
        if self.dataset is None:
            print("Dataset not loaded. Call load_dataset() first.")
            return None

        try:
            processed_examples = []
            for example in self.dataset[split]:
                # Get the text
                text = ' '.join(example['post_tokens'])

                # Get the label (hate speech or not)
                # In HateXplain, label is a list of numeric annotations
                # 0: hate speech, 1: neutral, 2: offensive
                labels = example['annotators']['label']

                # Count occurrences of each label
                label_counts = {}
                for label in labels:
                    label_counts[label] = label_counts.get(label, 0) + 1

                # Use the most common label
                most_common_label = max(label_counts.items(), key=lambda x: x[1])[0]

                # Get the target group
                target = example['annotators']['target']
                # If multiple targets, take the first one
                target_group = target[0] if target else 'None'

                processed_examples.append({
                    'text': text,
                    'labels': most_common_label,  # Using 'labels' to match the expected format
                    'target_group': target_group
                })

            # Update the dataset with processed examples
            dataset_processed = Dataset.from_list(processed_examples)
            print(f"Processed {len(processed_examples)} examples from {split} split")
            return dataset_processed
        except Exception as e:
            print(f"Error processing labels for {split} split: {str(e)}")
            raise

    def print_dataset_statistics(self):
        """
        Print statistics about the processed dataset including:
        - Number of examples per split
        - Label distribution (hate speech vs normal)
        - Target group distribution
        - Average text length
        """
        if self._processed_dataset is None:
            print(f"The dataset is not loaded")
            raise ValueError("The dataset is not loaded")

        print("\nDataset Statistics:")
        print("=" * 50)

        for split in self._processed_dataset.keys():
            print(f"\n{split.upper()} Split Statistics:")
            print("-" * 30)

            examples = self._processed_dataset[split]

            # Basic counts
            total_examples = len(examples)
            print(f"Total examples: {total_examples}")

            # Label distribution
            hate_speech_count = sum(1 for ex in examples if ((ex['labels'] == 0) or (ex['labels'] == 2)))  # Updated to use 'labels'
            normal_count = sum(1 for ex in examples if ex['labels'] == 1)  # Updated to use 'labels'
            print("\nLabel Distribution:")
            print(f"Hate Speech/Offensive: {hate_speech_count} ({hate_speech_count/total_examples*100:.2f}%)")
            print(f"Normal: {normal_count} ({normal_count/total_examples*100:.2f}%)")

            # Target group distribution
            target_groups = {}
            for ex in examples:
                target = ex['target_group']
                if isinstance(target, list):
                    target = target[0] if target else 'none'
                target_groups[target] = target_groups.get(target, 0) + 1

            print("\nTarget Group Distribution:")
            for target, count in sorted(target_groups.items(), key=lambda x: x[1], reverse=True):
                print(f"{target}: {count} ({count/total_examples*100:.2f}%)")

            # Text length statistics
            text_lengths = [len(ex['text'].split()) for ex in examples]
            avg_length = sum(text_lengths) / len(text_lengths)
            max_length = max(text_lengths)
            min_length = min(text_lengths)

            print("\nText Length Statistics:")
            print(f"Average length: {avg_length:.2f} words")
            print(f"Maximum length: {max_length} words")
            print(f"Minimum length: {min_length} words")

            # Hate speech by target group
            print("\nHate Speech Distribution by Target Group:")
            hate_by_target = {}
            for ex in examples:
                if ((ex['labels'] == 0) or (ex['labels'] == 2)):  # Updated to use 'labels'
                    target = ex['target_group']
                    if isinstance(target, list):
                        target = target[0] if target else 'none'
                    hate_by_target[target] = hate_by_target.get(target, 0) + 1

            for target, count in sorted(hate_by_target.items(), key=lambda x: x[1], reverse=True):
                total_for_target = target_groups[target]
                print(f"{target}: {count} hate speech examples ({count/total_for_target*100:.2f}% of this target)")

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
                target = example['target_group']
                if isinstance(target, list):
                    target = target[0] if target else 'none'
                new_example['target_group'] = self.standardization_map.get(
                    target,
                    target
                )
                new_examples.append(new_example)

            # Replace the split with the new standardized data
            self.dataset[split] = Dataset.from_list(new_examples)

    def remove_small_group(self) -> Dict[str, Dataset]:
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
            examples = self.dataset[split]

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
                    filtered_examples.append(example)

            # Create a new Dataset object from the filtered list
            new_dataset[split] = Dataset.from_list(filtered_examples)
            print(f"Removed {len(examples) - len(filtered_examples)} examples from {split} split.")
            print(f"Remaining examples in {split} split: {len(filtered_examples)}")

        self.dataset = new_dataset

        return new_dataset

    def get_dataset(self) -> Optional[Dataset]:
        """
        Get the processed dataset.

        Returns:
            The processed dataset or None if not loaded
        """
        return self.dataset