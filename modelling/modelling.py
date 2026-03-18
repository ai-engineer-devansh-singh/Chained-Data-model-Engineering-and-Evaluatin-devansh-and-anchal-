from model.randomforest import RandomForest


def model_predict(data, df, name):
    """
    Train and evaluate RandomForest model
    :param data: Data object containing train/test splits
    :param df: Original dataframe (not used in this version)
    :param name: Model name identifier
    """
    print(f"\n=== Training {name} Model ===")
    
    # Create RandomForest model instance
    model = RandomForest(
        model_name=name,
        embeddings=data.get_embeddings(),
        y=data.get_type()
    )
    
    # Train the model
    print("Training model...")
    model.train(data)
    
    # Make predictions
    print("Making predictions...")
    model.predict(data.get_X_test())
    
    # Evaluate and print results
    print("\n=== Evaluation Results ===")
    model_evaluate(model, data)


def model_evaluate(model, data):
    """
    Evaluate model performance
    :param model: Trained model instance
    :param data: Data object with test set
    """
    model.print_results(data)