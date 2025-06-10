# -*- coding: utf-8 -*-

# Portions of this code were written or refined with assistance from Cursor AI (https://www.cursor.sh)

"""
This module implements a fine-tunable Toxigen model for toxic content detection.

The module provides a Toxigen_Model class that wraps the tomh/toxigen_hateBERT model
and provides methods for training, evaluation, and prediction. It supports fine-tuning
with various training configurations and evaluation metrics.

Classes:
    Toxigen_Model: Main class for toxic content detection using Toxigen HateBERT
"""

__author__ = "Marc Bonhôte, Mikaël Schär, Viva Berlenghi"
__status__ = "Final"

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
    AdamW
)
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score
from typing import Dict, List, Optional, Union, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Toxigen_Model:
    def __init__(
        self,
        model_name: str = "tomh/toxigen_hateBERT",
        num_labels: int = 2,
        device: Optional[str] = None,
        max_length: int = 512
    ):
        """
        Initialize the Toxigen classifier.

        Args:
            model_name: Name of the pre-trained model to use (default: tomh/toxigen_hateBERT)
            num_labels: Number of classification labels (default: 2 for binary classification)
            device: Device to run the model on ('cuda' or 'cpu')
            max_length: Maximum sequence length for tokenization (default: 512)
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        
        # Set device
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize model and tokenizer
        self.tokenizer = None
        self.model = None
        self._initialize_model()
        
        # Training state
        self.optimizer = None
        self.scheduler = None
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': []
        }

    def _initialize_model(self):
        """
        Initialize the Toxigen model and tokenizer.
        Sets up the HateBERT-based model with appropriate configuration for toxic content detection.
        """
        try:
            # Initialize tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                model_max_length=self.max_length
            )
            
            # Initialize model with appropriate configuration
            if "hatebert" in self.model_name.lower():
                # For HateBERT models
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name,
                    num_labels=self.num_labels,
                    problem_type="single_label_classification"
                )
            else:
                # For other models
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name,
                    num_labels=self.num_labels
                )
            
            self.model.to(self.device)
            logger.info(f"Model {self.model_name} and tokenizer initialized successfully")
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'}) #ICI
            self.model.resize_token_embeddings(len(self.tokenizer)) # ICI
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise

    def prepare_inputs(
        self,
        texts: Union[str, List[str]],
        labels: Optional[Union[int, List[int]]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare inputs for the Toxigen model.

        Args:
            texts: Input text or list of texts to be classified for toxicity
            labels: Optional labels for training (0 for non-toxic, 1 for toxic)

        Returns:
            Dictionary of model inputs including tokenized text and optional labels
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize inputs with model-specific settings
        if "hatebert" in self.model_name.lower():
            # For HateBERT models
            encodings = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
        else:
            # For other models
            encodings = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
                add_special_tokens=True
            )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in encodings.items()}
        
        # Add labels if provided
        if labels is not None:
            if isinstance(labels, int):
                labels = [labels]
            inputs['labels'] = torch.tensor(labels).to(self.device)
        
        return inputs

    def train(
        self,
        train_dataloader: DataLoader,
        num_epochs: int = 3,
        learning_rate: float = 2e-5,
        warmup_steps: int = 0,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        save_path: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """
        Train the Toxigen model on toxic content detection.

        Args:
            train_dataloader: DataLoader containing training data with text and toxicity labels
            num_epochs: Number of training epochs (default: 3)
            learning_rate: Learning rate for optimizer (default: 2e-5)
            warmup_steps: Number of warmup steps for learning rate scheduler (default: 0)
            weight_decay: Weight decay for optimizer (default: 0.01)
            max_grad_norm: Maximum gradient norm for clipping (default: 1.0)
            save_path: Optional path to save the trained model

        Returns:
            Dictionary containing training metrics (loss and accuracy)
        """
        # Initialize metrics tracking
        metrics = {
            'train_loss': [],
            'train_accuracy': []
        }

        # Initialize optimizer and scheduler
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        total_steps = len(train_dataloader) * num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Training loop
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            
            # Training
            self.model.train()
            train_loss = 0
            train_preds = []
            train_labels = []
            
            progress_bar = tqdm(train_dataloader, desc="Training")
            for batch in progress_bar:
                # Prepare inputs
                inputs = self.prepare_inputs(
                    batch['text'],
                    batch['labels']
                )
                
                # Forward pass
                outputs = self.model(**inputs)
                loss = outputs.loss
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_grad_norm
                )
                
                # Optimizer step
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                # Update metrics
                train_loss += loss.item()
                train_preds.extend(torch.argmax(outputs.logits, dim=-1).cpu().numpy())
                train_labels.extend(inputs['labels'].cpu().numpy())
                
                # Update progress bar
                progress_bar.set_postfix({'loss': loss.item()})
            
            # Calculate training metrics
            train_loss /= len(train_dataloader)
            train_accuracy = accuracy_score(train_labels, train_preds)
            
            # Update metrics history
            metrics['train_loss'].append(train_loss)
            metrics['train_accuracy'].append(train_accuracy)
            
            logger.info(
                f"Epoch {epoch + 1} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Train Accuracy: {train_accuracy:.4f}"
            )
        
        # Save model if path provided
        if save_path:
            self.save_model(save_path)
        
        return metrics

    def predict(
        self,
        test_dataloader: DataLoader,
        return_probs: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Make predictions on text data using the trained Toxigen model.
        
        Args:
            test_dataloader: DataLoader containing test data or list of texts to classify
            return_probs: Whether to return probability scores along with predictions
            
        Returns:
            If return_probs is True: Tuple of (predictions, probabilities)
            If return_probs is False: Array of predictions (0 for non-toxic, 1 for toxic)
        """
        self.model.eval()
        predictions = []
        probabilities = []
        
        # Check if input is a list of strings
        if isinstance(test_dataloader, list):
            # Process in batches
            batch_size = 8
            for i in range(0, len(test_dataloader), batch_size):
                batch_texts = test_dataloader[i:i+batch_size]
                
                # Prepare inputs based on model type
                inputs = self.prepare_inputs(batch_texts)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    
                    # Handle different model output formats
                    if hasattr(outputs, 'logits'):
                        probs = torch.softmax(logits, dim=1)
                        batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
                        batch_probs = probs.cpu().numpy()
                    else:
                        # Some models might output probabilities directly
                        probs = outputs
                        batch_preds = torch.argmax(probs, dim=1).cpu().numpy()
                        batch_probs = probs.cpu().numpy()
                    
                    predictions.extend(batch_preds)
                    probabilities.extend(batch_probs)
        else:
            # Original DataLoader processing
            progress_bar = tqdm(test_dataloader, desc="Testing")
            
            with torch.no_grad():
                for batch in progress_bar:
                    try:
                        # Only pass text to prepare_inputs, not labels
                        inputs = self.prepare_inputs(batch['text'])
                        
                        outputs = self.model(**inputs)
                        logits = outputs.logits
                        probs = torch.softmax(logits, dim=1)
                        batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
                        batch_probs = probs.cpu().numpy()
                        
                        predictions.extend(batch_preds)
                        probabilities.extend(batch_probs)
                    except Exception as e:
                        logger.error(f"Error processing batch: {str(e)}")
                        continue
        
        predictions = np.array(predictions)
        probabilities = np.array(probabilities)
        
        if return_probs:
            return predictions, probabilities
        return predictions