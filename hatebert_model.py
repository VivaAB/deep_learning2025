# -*- coding: utf-8 -*-

"""
This module implements a fine-tunable HateBERT model for hate speech detection.

The module provides a HateBERTClassifier class that wraps the GroNLP/hateBERT model
and provides methods for training, evaluation, and prediction. It supports fine-tuning
with various training configurations and evaluation metrics.

Classes:
    HateBERTClassifier: Main class for hate speech detection using HateBERT
"""

__author__ = "Marc Bonhôte, Mikaël Schär, Viva ?"
__status__ = "Development"

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
from sklearn.metrics import accuracy_score, confusion_matrix
from typing import Dict, List, Optional, Union, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HateBERTClassifier:
    def __init__(
        self,
        model_name: str = "GroNLP/hateBERT",
        num_labels: int = 2,
        device: Optional[str] = None,
        max_length: int = 512
    ):
        """
        Initialize the HateBERT classifier.

        Args:
            model_name: Name of the pre-trained model to use
            num_labels: Number of classification labels
            device: Device to run the model on ('cuda' or 'cpu')
            max_length: Maximum sequence length for tokenization
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
        """Initialize the model and tokenizer."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                model_max_length=self.max_length
            )
            
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=self.num_labels
            )
            
            self.model.to(self.device)
            logger.info("Model and tokenizer initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise

    def prepare_inputs(
        self,
        texts: Union[str, List[str]],
        labels: Optional[Union[int, List[int]]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare inputs for the model.

        Args:
            texts: Input text or list of texts
            labels: Optional labels for training

        Returns:
            Dictionary of model inputs
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize inputs
        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
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
        Train the model.

        Args:
            train_dataloader: DataLoader for training data
            num_epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            warmup_steps: Number of warmup steps for learning rate scheduler
            weight_decay: Weight decay for optimizer
            max_grad_norm: Maximum gradient norm for clipping
            save_path: Optional path to save the model

        Returns:
            Dictionary containing training metrics
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
        Make predictions on new texts.

        Args:
            texts: Input text or list of texts
            return_probs: Whether to return probability scores

        Returns:
            Predictions or tuple of (predictions, probabilities)
        """
        self.model.eval()
        all_preds = []
        all_probs = []
        
        progress_bar = tqdm(test_dataloader, desc="Testing")
        with torch.no_grad():
            for batch in progress_bar:

                inputs = self.prepare_inputs(
                    batch['text'],
                    batch['labels']
                )

                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(logits, dim=-1)
                
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        if return_probs:
            return np.array(all_preds), np.array(all_probs)
        return np.array(all_preds)

    def save_model(self, path: str):
        """
        Save the model and tokenizer.

        Args:
            path: Path to save the model
        """
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        """
        Load a saved model and tokenizer.

        Args:
            path: Path to the saved model
        """
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model.to(self.device)
        logger.info(f"Model loaded from {path}")