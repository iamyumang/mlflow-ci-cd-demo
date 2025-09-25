import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def main():
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc:.4f}")

    # Log to MLflow (creates a run in mlruns if running locally)
    with mlflow.start_run():
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "mlflow_model")

    # Also save locally for Docker packaging
    mlflow.sklearn.save_model(model, "model")

if __name__ == "__main__":
    main()
