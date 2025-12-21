@echo off
mlflow server `
  --host 127.0.0.1 `
  --port 5000 `
  --backend-store-uri "file:///C:/Users/sobha/Desktop/detectron2/Code/Implement Detectron 2/mlflow/mlruns" `
  --default-artifact-root "file:///C:/Users/sobha/Desktop/detectron2/Code/Implement Detectron 2/mlflow/mlartifacts"


mlflow server `
  --host 127.0.0.1 `
  --port 5000 `
  --backend-store-uri "file:///C:/Users/sobha/Desktop/detectron2/Code/Implement Detectron 2/mlflow/mlruns" `
  --default-artifact-root "file:///C:/Users/sobha/Desktop/detectron2/Code/Implement Detectron 2/mlflow/mlartifacts"


mlflow server --host 127.0.0.1 --port 5000 `
  --backend-store-uri "sqlite:///mlflow/mlflow.db" `
  --default-artifact-root "file:///mlflow/mlartifacts"
