# Toxic Content Detection with Deep Learning

This repository contains the implementation of a deep learning-based toxic content detection system using the ToxiGen dataset. The project focuses on detecting toxic content while addressing algorithmic bias through various data augmentation techniques.

## Project Structure

```
├── data_augmentation.py      # Data augmentation techniques implementation
├── Toxigen_dataset.py        # ToxiGen dataset handler
├── Toxigen_model.py          # Toxic content detection model
├── main.ipynb                # Main notebook for experiments
├── requirements.txt          # Project dependencies
└── README.md                # Project documentation
```

## Features

- Toxic content detection using fine-tuned HateBERT model
- Multiple data augmentation techniques:
  - Back translation (English-French-English)
  - Synonym replacement using WordNet
  - Paraphrasing using T5
  - Random word deletion
- Comprehensive dataset analysis and visualization
- Bias mitigation through targeted data augmentation

## Setup

1. Clone the repository:
```bash
git clone [repository-url]
cd deep_learning2025
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download required NLTK data:
```python
import nltk
nltk.download('wordnet')
nltk.download('punkt')
```

## Model Architecture

The project uses a fine-tuned HateBERT model for toxic content detection:
- Base model: tomh/toxigen_hateBERT
- Task: Binary classification (toxic vs non-toxic)
- Input: Text sequences (max length: 512 tokens)
- Output: Binary classification with probability scores

## Data Augmentation Techniques

1. **Back Translation**
   - Translates text to French and back to English
   - Helps preserve meaning while introducing variation

2. **Synonym Replacement**
   - Uses WordNet to replace words with synonyms
   - Maintains semantic meaning while increasing diversity

3. **Paraphrasing**
   - Uses T5 model to generate paraphrases
   - Creates semantically equivalent but syntactically different versions

4. **Random Deletion**
   - Randomly removes words with probability p
   - Helps model learn to be robust to missing information

## Authors

- Marc Bonhôte
- Mikaël Schär
- Viva Berlenghi

## License

### Third-Party Assets

- **ToxiGen Dataset**: Used in accordance with its [license and usage terms](https://github.com/microsoft/ToxiGen#license). Please cite the original paper if using this data.

- **HateBERT Model**: Based on the implementation from [`tomh/toxigen_hateBERT`](https://github.com/tomh/toxigen_hateBERT), which is itself based on the HateBERT model by Caselli et al. (2021). Refer to the repository for licensing details.

---

### 🔖 Citations

If you use this project, please also cite the original datasets and models:

#### ToxiGen:
> Hartvigsen, T., Wallace, E., Singh, S., & Gardner, M. (2022). ToxiGen: A Large-Scale Machine-Generated Dataset for Adversarial and Implicit Hate Speech Detection. *ACL 2022*.

#### HateBERT:
> Caselli, T., Basile, V., Mitrović, J., & Hee, C. D. (2021). HateBERT: Retraining BERT for Abusive Language Detection in English. *Workshop on Abusive Language Online (ALW 2021)*.
