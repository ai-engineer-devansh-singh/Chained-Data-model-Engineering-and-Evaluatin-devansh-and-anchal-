import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from Config import *
from utils import *
import random

seed = 0
random.seed(seed)
np.random.seed(seed)


class ChainedData():
    """
    Data class for chained multi-label classification.
    Uses the SAME dataset and train/test split for all three levels:
    1. Type 2 only
    2. Type 2 + Type 3 combined
    3. Type 2 + Type 3 + Type 4 combined
    
    Only records with ALL three type columns present are used,
    ensuring consistent evaluation across all levels.
    """
    
    def __init__(self, X: np.ndarray, df: pd.DataFrame) -> None:
        print("\n=== Chained Data Encapsulation ===")
        
        # Filter to only records that have ALL type columns present
        df_filtered = filter_non_null(df.copy(), ['Type 2', 'Type 3', 'Type 4'])
        
        # Remove rare Type 2 classes
        df_filtered = remove_rare_classes(df_filtered, 'Type 2', min_instances=5)
        
        # Map filtered rows back to original X positions
        X_filtered = X[df_filtered.index.values]
        df_filtered = df_filtered.reset_index(drop=True)
        
        # Create three target levels (same emails, progressively specific labels)
        y_l1 = df_filtered['Type 2'].values
        y_l2 = (df_filtered['Type 2'].astype(str) + '_' + 
                df_filtered['Type 3'].astype(str)).values
        y_l3 = (df_filtered['Type 2'].astype(str) + '_' + 
                df_filtered['Type 3'].astype(str) + '_' + 
                df_filtered['Type 4'].astype(str)).values
        
        # ONE train/test split for ALL levels (stratified on Type 2 for balance)
        (self.X_train, self.X_test,
         self.y_train_l1, self.y_test_l1,
         self.y_train_l2, self.y_test_l2,
         self.y_train_l3, self.y_test_l3) = train_test_split(
            X_filtered, y_l1, y_l2, y_l3,
            test_size=0.2, random_state=seed, stratify=y_l1
        )
        
        # Print summary
        print(f"\nTotal records (with all types): {len(X_filtered)}")
        print(f"Training set: {len(self.X_train)} samples")
        print(f"Test set: {len(self.X_test)} samples")
        
        print(f"\nLevel 1 (Type 2 only):")
        print(f"  Classes: {len(np.unique(y_l1))}")
        print(f"  Unique labels: {list(np.unique(y_l1))}")
        
        print(f"\nLevel 2 (Type 2 + Type 3):")
        print(f"  Unique combinations: {len(np.unique(y_l2))}")
        
        print(f"\nLevel 3 (Type 2 + Type 3 + Type 4):")
        print(f"  Unique combinations: {len(np.unique(y_l3))}")
    
    # Shared X getters (same for all levels)
    def get_X_train(self):
        return self.X_train
    
    def get_X_test(self):
        return self.X_test
    
    # Level 1 y getters
    def get_y_train_l1(self):
        return self.y_train_l1
    
    def get_y_test_l1(self):
        return self.y_test_l1
    
    # Level 2 y getters
    def get_y_train_l2(self):
        return self.y_train_l2
    
    def get_y_test_l2(self):
        return self.y_test_l2
    
    # Level 3 y getters
    def get_y_train_l3(self):
        return self.y_train_l3
    
    def get_y_test_l3(self):
        return self.y_test_l3
