# Methods related to converting text into numeric representation and then returning numeric representation may go here
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from Config import Config


def get_tfidf_embd(df: pd.DataFrame):
    """
    Generate TF-IDF embeddings from text columns
    :param df: Input dataframe with text columns
    :return: TF-IDF matrix (numpy array)
    """
    # Combine Ticket Summary and Interaction Content
    # Handle NaN values by replacing with empty string
    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].fillna('')
    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].fillna('')
    
    # Combine both text columns
    combined_text = df[Config.TICKET_SUMMARY] + ' ' + df[Config.INTERACTION_CONTENT]
    
    # Initialize TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(
        max_features=5000,
        min_df=2,
        ngram_range=(1, 2),  # Use unigrams and bigrams
        stop_words='english',
        strip_accents='unicode'
    )
    
    # Fit and transform
    X = vectorizer.fit_transform(combined_text)
    
    print(f"TF-IDF Embeddings: Shape {X.shape}, Features {X.shape[1]}")
    
    # Convert sparse matrix to dense numpy array
    return X.toarray()