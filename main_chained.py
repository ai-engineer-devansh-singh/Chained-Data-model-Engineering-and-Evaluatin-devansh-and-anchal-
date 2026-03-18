# This is the main file for Chained Multi-Label Classification
# Design Decision 1: Chained Multi-Outputs

from preprocess import *
from embeddings import *
from modelling.chained_data_model import ChainedData
from model.chained import ChainedClassifier
import random
import numpy as np

seed = 0
random.seed(seed)
np.random.seed(seed)


def load_data():
    """Load the input data"""
    df = get_input_data()
    return df


def preprocess_data(df):
    """Preprocess the data"""
    # De-duplicate input data
    df = de_duplication(df)
    # Remove noise in input data
    df = noise_remover(df)
    # Translate data to english (placeholder)
    df[Config.TICKET_SUMMARY] = translate_to_en(df[Config.TICKET_SUMMARY].tolist())
    return df


def get_embeddings(df: pd.DataFrame):
    """Get TF-IDF embeddings"""
    X = get_tfidf_embd(df)  # get tf-idf embeddings
    return X, df


def get_chained_data_object(X: np.ndarray, df: pd.DataFrame):
    """Create ChainedData object with three label levels"""
    return ChainedData(X, df)


def perform_chained_modelling(chained_data: ChainedData):
    """Train and evaluate chained classifier"""
    # Create chained classifier
    model = ChainedClassifier(model_name='ChainedRandomForest')
    
    # Train all three levels
    model.train(chained_data)
    
    # Make predictions
    model.predict(chained_data)
    
    # Print results
    model.print_results(chained_data)


# Code will start executing from following line
if __name__ == '__main__':
    print("="*70)
    print("DESIGN DECISION 1: CHAINED MULTI-OUTPUT CLASSIFICATION")
    print("="*70)
    
    # Pre-processing steps
    df = load_data()
    df = preprocess_data(df)
    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].values.astype('U')
    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].values.astype('U')
    
    # Data transformation
    X, group_df = get_embeddings(df)
    
    # Create chained data model
    chained_data = get_chained_data_object(X, df)
    
    # Perform chained multi-label modelling
    perform_chained_modelling(chained_data)
    
    print("\n" + "="*70)
    print("CHAINED CLASSIFICATION COMPLETE")
    print("="*70)
