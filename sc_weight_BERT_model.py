from transformers import BertTokenizer, BertForSequenceClassification
import torch
from tqdm import tqdm
import evaluate
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

def load_sc_weighted_BERT():
    """
    Load the pre-trained SC-weighted BERT model and tokenizer.
    
    Returns:
        model: Pre-trained SC-weighted BERT model
        tokenizer: Tokenizer for the SC-weighted BERT model
    """
    # Load the pre-trained model and tokenizer
    model_name = "nom_du_modele/sc_weighted_BERT"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name)
    return model, tokenizer