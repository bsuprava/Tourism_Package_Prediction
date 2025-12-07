
# for data manipulation
import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

# for model training, tuning, and evaluation
import xgboost 
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score
)
# for model tracking
import mlflow

# for model serialization
import joblib

# for creating a folder
import os

# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError

# Connect to HuggingFace Space using token from git secret
print("Connecting to Huggingface...")
try:
    api = HfApi(token=os.getenv("HF_TOKEN"))
    print("Connected..")
except Exception as e:
    print(f"Error connecting to HuggingFace Space:{e}")

# 3.Retrieve train and test dataset from HuggingFace Space
print("Retrieve train and test dataset from HuggingFace Space.")
try:
    Xtrain_path = "hf://datasets/supravab/Tourism_Package_Prediction/X_train.csv"
    Xtest_path = "hf://datasets/supravab/Tourism_Package_Prediction/X_test.csv"
    ytrain_path = "hf://datasets/supravab/Tourism_Package_Prediction/y_train.csv"
    ytest_path = "hf://datasets/supravab/Tourism_Package_Prediction/y_test.csv"

    X_train = pd.read_csv(Xtrain_path)
    X_test = pd.read_csv(Xtest_path)
    y_train = pd.read_csv(ytrain_path)
    y_test = pd.read_csv(ytest_path)
    print("train and test dataset loaded successfully.")
except Exception as e:
    print(f"Error loading dataset:{e}")


# ===========================
# 1. XGBoost Model
# ===========================
numeric_features = X_train.select_dtypes(include=[np.number]).columns
categorical_features = X_train.select_dtypes(include=['object', 'category']).columns
# Set the clas weight to handle class imbalance
class_weight = y_train.value_counts()[0] / y_train.value_counts()[1]

tunedxgb = XGBClassifier(
    objective="binary:logistic", 
    random_state=1,
    eval_metric='logloss',
    tree_method="hist",
    scale_pos_weight=class_weight
)

# ===========================
# 3. Pipeline (Preprocessor + XGB)
# ===========================
print("Build xgbclassifier model with pipeline.")
xgb_preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (SimpleImputer(strategy="median"), numeric_features),
    (OneHotEncoder(drop='first',handle_unknown='ignore', sparse_output=False), categorical_features)       
)
xgb_pipe = Pipeline(steps=[
    ('preprocessor', xgb_preprocessor),
    ('xgbclassifier', tunedxgb)
])

# ===========================
# 4. MLflow Experiment
# ===========================
print("Connecting MLflow Url")
try:
  # Set the tracking URL for MLflow
  mlflow.set_tracking_uri("http://localhost:5000")

  # Set the name for the experiment
  mlflow.set_experiment("Tourism_Package_Prediction_Experiment1")
except Exception as e:
    print(f"Error connecting mlflow url:{e}")

print("Start MLflow Experiment Tracking")
best_model = None
try:
  
  with mlflow.start_run():
    xgb_pipe.fit(X_train, y_train)    
    best_model=xgb_pipe
    # ===========================
    # 5. Tune the Classification Threshold
    # ===========================
    y_prob1 = best_model.predict_proba(X_train)[:, 1]

    best_thr = 0.5
    best_f1 = 0

    for thr in np.arange(0.2, 0.8, 0.01):
        y_pred_thr = (y_prob1 >= thr).astype(int)
        score = f1_score(y_train, y_pred_thr)
        if score > best_f1:
            best_thr = thr
            best_f1 = score

    print(f"\nBest Threshold: {best_thr}")
    print(f"Best F1 Score at Threshold: {best_f1}")

    mlflow.log_metric("best_threshold", best_thr)
    mlflow.log_metric("best_threshold_f1", best_f1)

    # ===========================
    # 6. Evaluate Tuned Model
    # ===========================  
    
    y_pred_train_proba1 = best_tunedmodel.predict_proba(X_train)[:, 1]
    y_pred_train1 = (y_pred_train_proba1 >= best_thr).astype(int)

    y_pred_test_proba1 = best_tunedmodel.predict_proba(X_test)[:, 1]
    y_pred_test1 = (y_pred_test_proba1 >= best_thr).astype(int)

    train_report1 = classification_report(y_train, y_pred_train1, output_dict=True)
    test_report1 = classification_report(y_test, y_pred_test1, output_dict=True)

    metrics1 = {
    "train_accuracy": train_report1['accuracy'],
    "train_precision": train_report1['1']['precision'],
    "train_recall": train_report1['1']['recall'],
    "train_f1-score": train_report1['1']['f1-score'],
    "test_accuracy": test_report1['accuracy'],
    "test_precision": test_report1['1']['precision'],
    "test_recall": test_report1['1']['recall'],
    "test_f1-score": test_report1['1']['f1-score']
    }
    # print best tuning model scores
    print("=== LOGGED METRICS TO MLFLOW ===")
    for k, v in metrics1.items():
        print(f"{k}: {v}")

    mlflow.log_metrics(metrics1)
    print(f"model metrics logged in mlflow")
except Exception as e:
    print(f"Error in model training:{e}")

try:
  # Save the best model locally
  model_path = "tourism_package_prediction_modelv1.joblib"
  joblib.dump(best_model, model_path)

  # Log the model artifact
  mlflow.log_artifact(model_path, artifact_path="model")
  print(f"Model saved as artifact at: {model_path}")
except Exception as e:
  print(f"Unable to save the model:{e}")

# Upload joblib file to Hugging Face
print("Uploading joblib file to Hugging Face")
# Define Huggingface Repo Name and RepoType
repo_id = "supravab/Tourism_Package_Prediction"
repo_type = "model"

# Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Space '{repo_id}' created.")

# Upload joblib file of model to huggingface space
try:
    api.upload_file(
       path_or_fileobj="tourism_package_prediction_modelv1.joblib",
       path_in_repo="tourism_package_prediction_modelv1.joblib",
       repo_id=repo_id,
       repo_type=repo_type,
    ) 
    print(f"Joblib file uploaded: {model_path}")
except Exception as e:
    print(f"Error uploading joblib file:{e}")
