FROM python:3.10-slim

WORKDIR /app

# Install MLflow + scikit-learn
RUN pip install --no-cache-dir mlflow scikit-learn

# Copy model folder
COPY model /app/model

EXPOSE 1234

# Serve the MLflow model
CMD ["mlflow", "models", "serve", "-m", "/app/model", "-h", "0.0.0.0", "-p", "1234", "--env-manager", "local"]
