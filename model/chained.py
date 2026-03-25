import numpy as np
import pandas as pd
from model.base import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import random

seed = 0
np.random.seed(seed)
random.seed(seed)


class ChainedClassifier(BaseModel):
    """
    Chained Multi-Output Classifier
    Trains three separate RandomForest models for:
    1. Type 2 only
    2. Type 2 + Type 3 combined
    3. Type 2 + Type 3 + Type 4 combined
    """
    
    def __init__(self, model_name: str) -> None:
        super(ChainedClassifier, self).__init__()
        self.model_name = model_name

        # Three identical base estimators for three chain levels
        self.model_l1 = self._create_model()
        self.model_l2 = self._create_model()
        self.model_l3 = self._create_model()
        
        # Predictions storage
        self.predictions_l1 = None
        self.predictions_l2 = None
        self.predictions_l3 = None

    def _create_model(self):
        """Create the underlying estimator based on selected model_name."""
        model_key = self.model_name.lower().strip()

        if model_key in ["chainedrandomforest", "randomforest", "rf"]:
            return RandomForestClassifier(
                n_estimators=1000,
                random_state=seed,
                class_weight='balanced_subsample'
            )

        if model_key in ["svm", "chainedsvm"]:
            return SVC(kernel='rbf', C=1.0, random_state=seed, probability=True)

        if model_key in ["logistic", "logisticregression", "chainedlogistic"]:
            return LogisticRegression(
                max_iter=1000,
                random_state=seed,
                solver='lbfgs',
                multi_class='multinomial'
            )

        if model_key in ["gradientboost", "gradientboosting", "chainedgradientboost"]:
            return GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=seed
            )

        raise ValueError(
            f"Unsupported chain model '{self.model_name}'. "
            "Use one of: randomforest, svm, logistic, gradientboost"
        )
    
    def train(self, chained_data) -> None:
        """
        Train all three models on the same X_train with different y targets.
        :param chained_data: ChainedData object with shared train/test data
        """
        print("\n=== Training Chained Models ===")
        
        X_train = chained_data.get_X_train()
        
        # Train Level 1: Type 2 only
        print("\nTraining Level 1 (Type 2 only)...")
        self.model_l1.fit(X_train, chained_data.get_y_train_l1())
        
        # Train Level 2: Type 2 + Type 3
        print("Training Level 2 (Type 2 + Type 3)...")
        self.model_l2.fit(X_train, chained_data.get_y_train_l2())
        
        # Train Level 3: Type 2 + Type 3 + Type 4
        print("Training Level 3 (Type 2 + Type 3 + Type 4)...")
        self.model_l3.fit(X_train, chained_data.get_y_train_l3())
        
        print("All models trained successfully!")
    
    def predict(self, chained_data):
        """
        Make predictions using all three models on the same X_test.
        :param chained_data: ChainedData object with shared test data
        """
        print("\n=== Making Predictions ===")
        
        X_test = chained_data.get_X_test()
        
        # Predict all three levels on the same test set
        self.predictions_l1 = self.model_l1.predict(X_test)
        print("Level 1 predictions completed")
        
        self.predictions_l2 = self.model_l2.predict(X_test)
        print("Level 2 predictions completed")
        
        self.predictions_l3 = self.model_l3.predict(X_test)
        print("Level 3 predictions completed")
    
    def print_results(self, chained_data):
        """
        Print evaluation results for all three chain levels.
        Same test set used for all levels, showing cascading accuracy.
        :param chained_data: ChainedData object with ground truth
        """
        print("\n" + "="*70)
        print("CHAINED MULTI-LABEL CLASSIFICATION RESULTS")
        print("="*70)
        
        # Level 1 Results
        print("\n>>> LEVEL 1: Type 2 Classification <<<")
        print("-" * 70)
        y_true_l1 = chained_data.get_y_test_l1()
        accuracy_l1 = accuracy_score(y_true_l1, self.predictions_l1)
        print(f"Accuracy: {accuracy_l1:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_true_l1, self.predictions_l1, zero_division=0))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_true_l1, self.predictions_l1))
        
        # Level 2 Results
        print("\n>>> LEVEL 2: Type 2 + Type 3 Classification <<<")
        print("-" * 70)
        y_true_l2 = chained_data.get_y_test_l2()
        accuracy_l2 = accuracy_score(y_true_l2, self.predictions_l2)
        print(f"Accuracy: {accuracy_l2:.4f}")
        print(f"Accuracy Drop from Level 1: {(accuracy_l1 - accuracy_l2):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_true_l2, self.predictions_l2, zero_division=0))
        
        # Show top combinations
        unique, counts = np.unique(y_true_l2, return_counts=True)
        top_combinations = sorted(zip(unique, counts), key=lambda x: x[1], reverse=True)[:10]
        print(f"\nTop Label Combinations (out of {len(unique)} unique):")
        for combo, count in top_combinations:
            print(f"  {combo}: {count}")
        
        # Level 3 Results
        print("\n>>> LEVEL 3: Type 2 + Type 3 + Type 4 Classification <<<")
        print("-" * 70)
        y_true_l3 = chained_data.get_y_test_l3()
        accuracy_l3 = accuracy_score(y_true_l3, self.predictions_l3)
        print(f"Accuracy: {accuracy_l3:.4f}")
        print(f"Accuracy Drop from Level 2: {(accuracy_l2 - accuracy_l3):.4f}")
        print(f"Total Accuracy Drop from Level 1: {(accuracy_l1 - accuracy_l3):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_true_l3, self.predictions_l3, zero_division=0))
        
        # Show top combinations
        unique, counts = np.unique(y_true_l3, return_counts=True)
        top_combinations = sorted(zip(unique, counts), key=lambda x: x[1], reverse=True)[:10]
        print(f"\nTop Label Combinations (out of {len(unique)} unique):")
        for combo, count in top_combinations:
            print(f"  {combo}: {count}")
        
        # Summary
        print("\n" + "="*70)
        print("SUMMARY: Cascading Accuracy Analysis")
        print("="*70)
        print(f"Level 1 (Type 2):              {accuracy_l1:.4f}")
        print(f"Level 2 (Type 2+3):            {accuracy_l2:.4f}  (drop: {(accuracy_l1-accuracy_l2):.4f})")
        print(f"Level 3 (Type 2+3+4):          {accuracy_l3:.4f}  (drop: {(accuracy_l1-accuracy_l3):.4f})")
        print("="*70)
        print("\nNote: Accuracy degradation is expected as label combinations increase.")
        print("="*70)
    
    def data_transform(self) -> None:
        """Not used in this implementation"""
        pass
