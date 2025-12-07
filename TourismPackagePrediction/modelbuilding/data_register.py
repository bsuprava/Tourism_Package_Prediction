
# Import necessary libraries
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os

# Define Huggingface Repo Name and RepoType
repo_id = "supravab/Tourism_Package_Prediction"
repo_type = "dataset"

# Initialize API client using huggingface token from git secret
api = HfApi(token=os.getenv("HF_TOKEN"))

# Check if the space exists orelse create new space
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Space '{repo_id}' created.")

# Upload data file to huggingface space
try:
    api.upload_folder(
        folder_path="TourismPackagePrediction/data",
        repo_id=repo_id,
        repo_type=repo_type,
    )
except Exception as e:
    print(f"Error uploading dataset:{e}")
