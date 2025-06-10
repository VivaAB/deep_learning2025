# Toxic Content Detection with Deep Learning

This repository implements a deep learning-based toxic content detection system using the HateXplain and ToxiGen dataset with hateBERT models. The project focuses on detecting toxic content while addressing algorithmic bias through various data augmentation techniques.

## Project Structure

```
├── Toxigen_dataset.py                      # ToxiGen dataset handler
├── Toxigen_model.py                        # ToxiGen model implementation
├── data_augmentation.py                    # Data augmentation techniques implementation
├── run_ToxiGen.ipynb                       # Notebook for running ToxiGen experiments
├
├── HateXplain_dataset.py                   # HateXplain dataset handler
├── HateXplain_model.py                     # HateXplain model implementation
├── utils.py                                # Utility functions             
├── run_HateXplain.ipynb                    # Notebook for running HateXplain experiments
├
├── requirements.txt                        # Project dependencies
└── README.md                               # Project documentation
```

## Features

- Toxic content detection using fine-tuned HateBERT model
- Comprehensive dataset analysis and visualization
- Bias mitigation through targeted data augmentation
- Multiple data augmentation techniques:
  - Back translation
  - Synonym replacement
  - Paraphrasing
  - Random word deletion

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
5. In the run_ToxiGen.ipynb file, add you HuggingFace token :
```python
# Initialize and load the ToxiGen dataset
HUGGING_FACE_TOKEN = "your token here"
```

6. Run the code:
For Toxigen experiments:
```bash
jupyter notebook run_Toxigen.ipynb
```
For HateXplain experiments:
```bash
jupyter notebook run_HateXplain.ipynb
```
## Authors

- Marc Bonhôte
- Mikaël Schär
- Viva Berlenghi

## License

- **ToxiGen Dataset**: Used in accordance with its [license and usage terms](https://github.com/microsoft/ToxiGen#license). Please cite the original paper if using this data.

- **HateBERT Model**: Based on the implementation from [`tomh/toxigen_hateBERT`](https://github.com/tomh/toxigen_hateBERT), which is itself based on the HateBERT model by Caselli et al. (2021). Refer to the repository for licensing details.

- Code assistance provided by [Cursor AI](https://www.cursor.sh).
---

### 🔖 Citations

If you use this project, please also cite the original datasets and models:

#### ToxiGen:
> Hartvigsen, T., Wallace, E., Singh, S., & Gardner, M. (2022). ToxiGen: A Large-Scale Machine-Generated Dataset for Adversarial and Implicit Hate Speech Detection. *ACL 2022*.

#### HateBERT:
> Caselli, T., Basile, V., Mitrović, J., & Hee, C. D. (2021). HateBERT: Retraining BERT for Abusive Language Detection in English. *Workshop on Abusive Language Online (ALW 2021)*.
