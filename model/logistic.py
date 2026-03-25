import numpy as np
import pandas as pd
from model.base import BaseModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import random

seed = 0
np.random.seed(seed)
random.seed(seed)


class LogisticReg(BaseModel):
    """
    Logistic Regression implementation for multi-class classification
    """
    def __init__(self,
                 model_name: str,
                 embeddings: np.ndarray,
                 y: np.ndarray) -> None:
        super(LogisticReg, self).__init__()
        self.model_name = model_name
        self.embeddings = embeddings
        self.y = y
        self.mdl = LogisticRegression(
            max_iter=1000, 
            random_state=seed, 
            solver='lbfgs',
            multi_class='multinomial'
        )
        self.predictions = None
        self.data_transform()

    def train(self, data) -> None:
        print(f"Training {self.model_name}...")
        self.mdl = self.mdl.fit(data.X_train, data.y_train)

    def predict(self, X_test: pd.Series):
        self.predictions = self.mdl.predict(X_test)

    def print_results(self, data):
        accuracy = accuracy_score(data.y_test, self.predictions)
        print(f"\n{self.model_name} Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(data.y_test, self.predictions, zero_division=0))

    def data_transform(self) -> None:
        pass
