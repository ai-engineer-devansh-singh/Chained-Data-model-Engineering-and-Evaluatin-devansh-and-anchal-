# Methods related to data loading and all pre-processing steps will go here
import pandas as pd
import re
from Config import Config
import os


def get_input_data():
    """
    Load and combine data from both CSV files (AppGallery.csv and Purchasing.csv)
    :return: Combined dataframe
    """
    # Get the directory where this script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, 'data')
    
    # Load both datasets
    df_app = pd.read_csv(os.path.join(data_dir, 'AppGallery.csv'))
    df_purchasing = pd.read_csv(os.path.join(data_dir, 'Purchasing.csv'))
    
    # Combine datasets
    df = pd.concat([df_app, df_purchasing], ignore_index=True)
    
    print(f"Loaded {len(df_app)} records from AppGallery.csv")
    print(f"Loaded {len(df_purchasing)} records from Purchasing.csv")
    print(f"Total records: {len(df)}")
    
    return df


def de_duplication(df: pd.DataFrame):
    """
    Remove duplicate records based on Ticket Summary and Interaction content
    :param df: Input dataframe
    :return: Deduplicated dataframe
    """
    initial_count = len(df)
    
    # Drop duplicates based on ticket summary and interaction content
    df = df.drop_duplicates(subset=[Config.TICKET_SUMMARY, Config.INTERACTION_CONTENT], keep='first')
    
    final_count = len(df)
    removed = initial_count - final_count
    
    print(f"Deduplication: Removed {removed} duplicates, {final_count} records remain")
    
    return df.reset_index(drop=True)


def noise_remover(df: pd.DataFrame):
    """
    Remove noise from text columns (special characters, excess whitespace, etc.)
    :param df: Input dataframe
    :return: Cleaned dataframe
    """
    def clean_text(text):
        if pd.isna(text) or text == '':
            return ''
        
        # Convert to string
        text = str(text)
        
        # Remove HTML entities
        text = re.sub(r'&[a-z]+;', ' ', text)
        
        # Remove email addresses (keep content but remove email pattern)
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Remove phone numbers (various formats)
        text = re.sub(r'\+?\d[\d\s\-\(\)]{7,}\d', '', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s\.\,\!\?\-]', ' ', text)
        
        # Remove excess whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    # Clean text columns
    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].apply(clean_text)
    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].apply(clean_text)
    
    print("Noise removal: Text columns cleaned")
    
    return df


def translate_to_en(text_list):
    """
    Translate text to English (placeholder - keeping as is since translation is complex)
    For this implementation, we'll work with multilingual text as-is
    :param text_list: List of text strings
    :return: List of translated text (currently returns original)
    """
    # Note: For production, you would integrate a translation API (e.g., Google Translate)
    # For this assignment, we'll work with the text as-is
    print("Translation: Keeping original text (multilingual support)")
    return text_list
    
    return text_list