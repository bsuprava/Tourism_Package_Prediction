from huggingface_hub import HfApi
import os

# Connect to HuggingFace Space using token from git secret
print("Connecting to Huggingface...")
try:
    api = HfApi(token=os.getenv("HF_TOKEN"))
    print("Connected..")
except Exception as e:
    print(f"Error connecting to HuggingFace Space:{e}")


# Upload deployment files to huggingface space
print("Uploading deployment files to Huggingface...")
try:
    api.upload_folder(
       folder_path="TourismPackagePrediction/deployment",     # the local folder containing your files
       repo_id="supravab/Tourism_Package_Prediction",          # the target repo
       repo_type="space",                      # dataset, model, or space
       path_in_repo="",                          # optional: subfolder path inside the repo
    )
    print("Successfully uploaded deployment files to Huggingface.")
except Exception as e:
    print(f"Error uploading deployment files:{e}")
