# Chained Data Model Engineering

This is a small project for ticket text classification.
It uses TF-IDF and machine learning models.

You can do 3 things:
- Run a basic model
- Compare many models
- Run a chained model for Type 2, Type 3, and Type 4

## Quick Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas scikit-learn
```

## How to Run

Run basic model:
```bash
python main.py
```

Compare all models:
```bash
python main_comparison.py
```

Run chained model with RandomForest:
```bash
python main_chained.py --model randomforest
```

Run chained model with other options:
```bash
python main_chained.py --model svm
python main_chained.py --model logistic
python main_chained.py --model gradientboost
```

## Data Files

Input files are in the data folder:
- data/AppGallery.csv
- data/Purchasing.csv

## Output File

Comparison result is saved in:
- model_comparison_results.txt
