# Any extra functionality that need to be reused will go here
import pandas as pd
import numpy as np


def remove_rare_classes(df: pd.DataFrame, target_col: str, min_instances: int = 5):
    """
    Remove classes that have fewer than min_instances
    :param df: Input dataframe
    :param target_col: Name of the target column
    :param min_instances: Minimum number of instances required
    :return: Filtered dataframe
    """
    initial_count = len(df)
    
    # Count instances per class
    class_counts = df[target_col].value_counts()
    
    # Find classes with sufficient instances
    valid_classes = class_counts[class_counts >= min_instances].index
    
    # Filter dataframe
    df = df[df[target_col].isin(valid_classes)]
    
    final_count = len(df)
    removed = initial_count - final_count
    
    if removed > 0:
        print(f"Rare class removal: Removed {removed} instances from {target_col}, {final_count} records remain")
    
    return df


def filter_non_null(df: pd.DataFrame, columns: list):
    """
    Filter out rows where any of the specified columns have null values
    :param df: Input dataframe
    :param columns: List of column names to check
    :return: Filtered dataframe
    """
    initial_count = len(df)
    
    # Remove rows with null values in specified columns
    df = df.dropna(subset=columns)
    
    final_count = len(df)
    removed = initial_count - final_count
    
    if removed > 0:
        print(f"Null filtering: Removed {removed} records with null values, {final_count} records remain")
    
    return df