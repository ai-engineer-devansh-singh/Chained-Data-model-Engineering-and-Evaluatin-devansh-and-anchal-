import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from Config import *
from utils import *
import random
seed = 0
random.seed(seed)
np.random.seed(seed)

class Data():
    def __init__(self,
                 X: np.ndarray,
                 df: pd.DataFrame) -> None:
        # This method will create the model for data
        # This will be performed in second activity
        
        print("\n=== Data Encapsulation ===")
        
        # Store original embeddings and dataframe
        self.embeddings = X
        self.df = df
        
        # Remove records with null values in Type 2 (our main target)
        df = filter_non_null(df, [Config.CLASS_COL])
        
        # Remove rare classes (less than 5 instances)
        df = remove_rare_classes(df, Config.CLASS_COL, min_instances=5)
        
        # Extract target variable (y2 - Type 2)
        self.y = df[Config.CLASS_COL].values
        
        # Map filtered rows back to original X positions
        # (index values preserved since utils don't reset index)
        X_filtered = X[df.index.values]
        df = df.reset_index(drop=True)
        
        # Split data into train and test
        self.X_train, self.X_test, self.y_train, self.y_test, self.train_df, self.test_df = \
            train_test_split(X_filtered, self.y, df, test_size=0.2, random_state=seed, stratify=self.y)
        
        print(f"Training set: {len(self.X_train)} samples")
        print(f"Test set: {len(self.X_test)} samples")
        print(f"Number of classes: {len(np.unique(self.y))}")
        print(f"Classes: {np.unique(self.y)}")
        print(f"Class distribution in training:")
        unique, counts = np.unique(self.y_train, return_counts=True)
        for cls, cnt in zip(unique, counts):
            print(f"  {cls}: {cnt}")

    def get_type(self):
        return self.y
    
    def get_X_train(self):
        return self.X_train
    
    def get_X_test(self):
        return self.X_test
    
    def get_type_y_train(self):
        return self.y_train
    
    def get_type_y_test(self):
        return self.y_test
    
    def get_train_df(self):
        return self.train_df
    
    def get_embeddings(self):
        return self.embeddings
    
    def get_type_test_df(self):
        return self.test_df


