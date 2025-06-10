"""
This module implements text augmentation techniques for improving model training.

The module provides a DataAugmentor class that implements multiple augmentation strategies:
- Back translation (English-French-English)
- Synonym replacement using WordNet
- Paraphrasing using T5
- Random word deletion

These techniques can be used individually or in combination to create augmented versions
of training data, particularly useful for improving model performance on underrepresented classes.

Classes:
    DataAugmentor: Main class for implementing text augmentation techniques
"""

__author__ = "Marc Bonhôte, Mikaël Schär, Viva Berlenghi"
__status__ = "Final"

import torch
import nltk
from nltk.corpus import wordnet
from transformers import MarianMTModel, MarianTokenizer, T5ForConditionalGeneration, T5Tokenizer
import random
import logging
from typing import List, Dict, Optional
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataAugmentor:
    def __init__(
        self,
        device: Optional[str] = None,
        use_back_translation: bool = False,
        use_synonym_replacement: bool = False,
        use_paraphrasing: bool = False,
        use_random_deletion: bool = False
    ):
        """
        Initialize the DataAugmentor with various augmentation techniques.
        
        Args:
            device: Device to run models on ('cuda' or 'cpu')
            use_back_translation: Whether to use back-translation
            use_synonym_replacement: Whether to use synonym replacement
            use_paraphrasing: Whether to use paraphrasing
            use_random_deletion: Whether to use random deletion
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

        # Download required NLTK data
        try:
            nltk.data.find('corpora/wordnet')
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('wordnet')
            nltk.download('punkt')

        # Initialize back-translation models if enabled
        if use_back_translation:
            try:
                logger.info("Initializing translation models...")
                self.en_fr_model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-fr').to(self.device)
                self.fr_en_model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-fr-en').to(self.device)
                self.en_fr_tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-fr')
                self.fr_en_tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-fr-en')
                logger.info("Translation models initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing translation models: {str(e)}")
                use_back_translation = False

        # Initialize paraphrasing model if enabled
        if use_paraphrasing:
            try:
                logger.info("Initializing T5 model...")
                self.t5_model = T5ForConditionalGeneration.from_pretrained('t5-small').to(self.device)
                self.t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')
                logger.info("T5 model initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing T5 model: {str(e)}")
                use_paraphrasing = False

        self.use_back_translation = use_back_translation
        self.use_synonym_replacement = use_synonym_replacement
        self.use_paraphrasing = use_paraphrasing
        self.use_random_deletion = use_random_deletion

    def get_synonyms(self, word: str) -> List[str]:
        """
        Get synonyms for a word using WordNet.
        
        Args:
            word: The word to get synonyms for
            
        Returns:
            List of synonyms from WordNet
        """
        synonyms = []
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ')
                if synonym != word and synonym not in synonyms:
                    synonyms.append(synonym)
        return synonyms

    def synonym_replacement(self, text: str, n: int = 1) -> str:
        """
        Replace n random words with their synonyms.
        
        Args:
            text: The input text
            n: Number of words to replace
            
        Returns:
            Text with replaced synonyms
        """
        words = text.split()
        n = min(n, len(words))
        new_words = words.copy()

        random_word_list = list(set([word for word in words if len(word) > 3]))
        random.shuffle(random_word_list)

        num_replaced = 0
        for random_word in random_word_list:
            synonyms = self.get_synonyms(random_word)
            if len(synonyms) > 0:
                synonym = random.choice(synonyms)
                new_words = [synonym if word == random_word else word for word in new_words]
                num_replaced += 1
            if num_replaced >= n:
                break

        return ' '.join(new_words)

    def back_translate(self, text: str) -> str:
        """
        Translate text to French and back to English.
        
        Args:
            text: The input text
            
        Returns:
            Back-translated text, or original text if translation fails
        """
        try:
            # English to French
            inputs = self.en_fr_tokenizer(text, return_tensors="pt", padding=True).to(self.device)
            translated = self.en_fr_model.generate(**inputs)
            french = self.en_fr_tokenizer.decode(translated[0], skip_special_tokens=True)

            # French to English
            inputs = self.fr_en_tokenizer(french, return_tensors="pt", padding=True).to(self.device)
            translated = self.fr_en_model.generate(**inputs)
            english = self.fr_en_tokenizer.decode(translated[0], skip_special_tokens=True)

            return english
        except Exception as e:
            logger.error(f"Error in back translation: {str(e)}")
            return text

    def paraphrase(self, text: str) -> str:
        """
        Generate a paraphrase using T5.
        
        Args:
            text: The input text
            
        Returns:
            Paraphrased text, or original text if paraphrasing fails
        """
        try:
            input_text = f"paraphrase: {text}"
            inputs = self.t5_tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True).to(self.device)
            outputs = self.t5_model.generate(
                inputs,
                max_length=512,
                num_beams=4,
                no_repeat_ngram_size=2,
                early_stopping=True
            )
            return self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            logger.error(f"Error in paraphrasing: {str(e)}")
            return text

    def random_deletion(self, text: str, p: float = 0.1) -> str:
        """
        Randomly delete words from the text with probability p.
        
        Args:
            text: The input text
            p: Probability of deleting each word
            
        Returns:
            Text with randomly deleted words
        """
        try:
            # Tokenize the text
            words = text.split()

            # Keep words with probability 1-p
            new_words = [word for word in words if random.random() > p]

            # If all words were deleted, keep at least one word
            if len(new_words) == 0:
                new_words = [random.choice(words)]

            return ' '.join(new_words)
        except Exception as e:
            logger.error(f"Error in random deletion: {str(e)}")
            return text

    def augment_text(self, text: str) -> List[str]:
        """
        Apply all enabled augmentation techniques to the text.
        
        Args:
            text: The input text
            
        Returns:
            List of augmented versions of the text
        """
        augmented_texts = []

        if self.use_back_translation:
            augmented_texts.append(self.back_translate(text))

        if self.use_synonym_replacement:
            # Use both regular and WSD-based synonym replacement
            augmented_texts.append(self.synonym_replacement(text, n=2))

        if self.use_paraphrasing:
            augmented_texts.append(self.paraphrase(text))

        if self.use_random_deletion:
            # Apply random deletion with different probabilities
            augmented_texts.append(self.random_deletion(text, p=0.2))

        return augmented_texts

    def augment_dataset(
        self,
        dataset: Dict[str, List],
        target_groups: Optional[List[str]] = None,
        augmentation_factor: int = 1
    ) -> Dict[str, List]:
        """
        Augment the dataset for specified target groups.
        
        Args:
            dataset: Dictionary containing 'train' and 'test' datasets
            target_groups: List of target groups to augment
            augmentation_factor: Number of augmented versions to create per example
            
        Returns:
            Dictionary containing augmented datasets
        """
        augmented_data = {
            'train': {
                'text': [],
                'labels': [],
                'target_group': []
            },
            'test': {
                'text': [],
                'labels': [],
                'target_group': []
            }
        }

        # Process each split
        for split in ['train', 'test']:
            logger.info(f"\nProcessing {split} split...")

            # First, add all original examples
            logger.info("Adding original examples...")
            for i in tqdm(range(len(dataset[split]['text'])), desc="Original examples"):
                text = dataset[split]['text'][i]
                label = dataset[split]['labels'][i]
                target_group = dataset[split]['target_group'][i]

                augmented_data[split]['text'].append(text)
                augmented_data[split]['labels'].append(label)
                augmented_data[split]['target_group'].append(target_group)

            # Then, augment examples for target groups
            logger.info("Generating augmented examples...")
            texts_to_augment = []
            indices_to_augment = []

            # Collect texts that need augmentation
            for i in range(len(dataset[split]['text'])):
                if target_groups is None or dataset[split]['target_group'][i] in target_groups:
                    texts_to_augment.append(dataset[split]['text'][i])
                    indices_to_augment.append(i)

            # Process augmentation in batches
            batch_size = 8
            for i in tqdm(range(0, len(texts_to_augment), batch_size), desc="Augmentation batches"):
                batch_texts = texts_to_augment[i:i + batch_size]
                batch_indices = indices_to_augment[i:i + batch_size]

                for text, idx in zip(batch_texts, batch_indices):
                    for _ in range(augmentation_factor):
                        augmented_texts = self.augment_text(text)
                        for aug_text in augmented_texts:
                            augmented_data[split]['text'].append(aug_text)
                            augmented_data[split]['labels'].append(dataset[split]['labels'][idx])
                            augmented_data[split]['target_group'].append(dataset[split]['target_group'][idx])

            # Print statistics
            original_count = len(dataset[split]['text'])
            augmented_count = len(augmented_data[split]['text']) - original_count
            logger.info(f"\n{split.capitalize()} split statistics:")
            logger.info(f"Original examples: {original_count}")
            logger.info(f"Augmented examples: {augmented_count}")
            logger.info(f"Total examples: {len(augmented_data[split]['text'])}")

            # Print target group distribution
            if target_groups:
                logger.info("\nTarget group distribution:")
                for group in target_groups:
                    count = augmented_data[split]['target_group'].count(group)
                    logger.info(f"{group}: {count} examples")

        return augmented_data