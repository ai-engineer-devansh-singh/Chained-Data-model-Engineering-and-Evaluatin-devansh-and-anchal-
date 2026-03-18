"""
Multi-Model Comparison Script
Compares RandomForest, SVM, Logistic Regression, and Gradient Boosting
on the baseline Type 2 classification task
"""

from preprocess import *
from embeddings import *
from modelling.data_model import *
from model.randomforest import RandomForest
from model.svm import SVM
from model.logistic import LogisticReg
from model.gradientboost import GradientBoost
from sklearn.metrics import accuracy_score
import random
import time

seed = 0
random.seed(seed)
np.random.seed(seed)


def load_data():
    """Load the input data"""
    df = get_input_data()
    return df


def preprocess_data(df):
    """Preprocess the data"""
    df = de_duplication(df)
    df = noise_remover(df)
    df[Config.TICKET_SUMMARY] = translate_to_en(df[Config.TICKET_SUMMARY].tolist())
    return df


def get_embeddings(df: pd.DataFrame):
    """Get TF-IDF embeddings"""
    X = get_tfidf_embd(df)
    return X, df


def get_data_object(X: np.ndarray, df: pd.DataFrame):
    """Create Data object"""
    return Data(X, df)


def train_and_evaluate_model(model_class, model_name, data):
    """
    Train and evaluate a single model
    Returns: (model_name, accuracy, training_time)
    """
    print(f"\n{'='*70}")
    print(f"EVALUATING: {model_name}")
    print(f"{'='*70}")
    
    # Create model instance
    model = model_class(
        model_name=model_name,
        embeddings=data.get_embeddings(),
        y=data.get_type()
    )
    
    # Train and time
    start_time = time.time()
    model.train(data)
    training_time = time.time() - start_time
    
    # Predict
    model.predict(data.get_X_test())
    
    # Evaluate
    accuracy = accuracy_score(data.get_type_y_test(), model.predictions)
    model.print_results(data)
    
    print(f"\nTraining Time: {training_time:.2f} seconds")
    
    return model_name, accuracy, training_time


def main():
    print("="*70)
    print("MULTI-MODEL COMPARISON: BASELINE TYPE 2 CLASSIFICATION")
    print("="*70)
    
    # Preprocessing
    df = load_data()
    df = preprocess_data(df)
    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].values.astype('U')
    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].values.astype('U')
    
    # Data transformation
    X, group_df = get_embeddings(df)
    data = get_data_object(X, df)
    
    # Models to compare
    models = [
        (RandomForest, "RandomForest"),
        (SVM, "Support Vector Machine"),
        (LogisticReg, "Logistic Regression"),
        (GradientBoost, "Gradient Boosting")
    ]
    
    # Train and evaluate all models
    results = []
    for model_class, model_name in models:
        try:
            name, accuracy, train_time = train_and_evaluate_model(
                model_class, model_name, data
            )
            results.append((name, accuracy, train_time))
        except Exception as e:
            print(f"\nError with {model_name}: {e}")
            results.append((model_name, 0.0, 0.0))
    
    # Summary comparison
    print("\n" + "="*70)
    print("SUMMARY: MODEL COMPARISON")
    print("="*70)
    print(f"{'Model':<25} {'Accuracy':<15} {'Training Time (s)':<20}")
    print("-"*70)
    
    for name, accuracy, train_time in results:
        print(f"{name:<25} {accuracy:>6.4f} ({accuracy*100:>5.2f}%)  {train_time:>10.2f}")
    
    # Best model
    best_model = max(results, key=lambda x: x[1])
    print("\n" + "="*70)
    print(f"BEST MODEL: {best_model[0]} with {best_model[1]:.4f} accuracy")
    print("="*70)
    
    # Save results to file
    with open('model_comparison_results.txt', 'w') as f:
        f.write("="*70 + "\n")
        f.write("MULTI-MODEL COMPARISON RESULTS\n")
        f.write("Baseline Type 2 Classification\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Dataset: {len(data.get_X_train())} train, {len(data.get_X_test())} test samples\n")
        f.write(f"Classes: {len(np.unique(data.get_type()))}\n\n")
        
        f.write(f"{'Model':<25} {'Accuracy':<15} {'Training Time (s)':<20}\n")
        f.write("-"*70 + "\n")
        
        for name, accuracy, train_time in results:
            f.write(f"{name:<25} {accuracy:>6.4f} ({accuracy*100:>5.2f}%)  {train_time:>10.2f}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write(f"BEST MODEL: {best_model[0]} with {best_model[1]:.4f} accuracy\n")
        f.write("="*70 + "\n")
    
    print("\n✓ Results saved to: model_comparison_results.txt")


if __name__ == '__main__':
    main()
