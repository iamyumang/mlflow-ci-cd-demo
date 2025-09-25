import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import shutil

# Force MLflow to log to a safe folder on CI runners
mlflow.set_tracking_uri("file:///tmp/mlruns")

def main():
    # Load dataset
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc:.4f}")

    # Log MLflow run (metrics + model)
    with mlflow.start_run():
        mlflow.log_metric("accuracy", acc)

        # Optional: add signature & input example to remove warnings
        import pandas as pd
        from mlflow.models.signature import infer_signature

        input_example = pd.DataFrame(X_train[:5], columns=["sepal length", "sepal width", "petal length", "petal width"])
        signature = infer_signature(X_train, model.predict(X_train))

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="mlflow_model",
            signature=signature,
            input_example=input_example
        )

    # Save local model folder for Docker
    if os.path.exists("model"):
        shutil.rmtree("model")
    mlflow.sklearn.save_model(model, "model")

if __name__ == "__main__":
    main()
