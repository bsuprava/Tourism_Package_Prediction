# 1. Import necessary libraries
# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder

# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi


# 2. Connect to HuggingFace Space using token from git secret
try:
    api = HfApi(token=os.getenv("HF_TOKEN"))
except Exception as e:
    print(f"Error connecting to HuggingFace Space:{e}")

    
# 3.Retrieve dataset from HuggingFace Space
try:
    DATASET_PATH = "hf://datasets/supravab/Tourism_Package_Prediction/tourism.csv"
    tourismdf = pd.read_csv(DATASET_PATH)
    print("Dataset loaded successfully.")
except Exception as e:
    print(f"Error loading dataset:{e}")

# 4.Create a copy of Data
tdf = tourismdf.copy()

# 5.Drop the unique identifier
columns_to_drop = ['CustomerID', 'Unnamed: 0']
# Check which columns actually exist before dropping to avoid errors
existing_columns_to_drop = [col for col in columns_to_drop if col in tdf.columns]
if existing_columns_to_drop:
    tdf = tdf.drop(columns=existing_columns_to_drop)

# Handle missing values: Fill numerical columns with median/mean

# 6.Handle duplicates
tdf.drop_duplicates(inplace=True)

# 7.Treat Category column having equivalent data values as per initial dataset
# 7.1. Map the specific 'Fe Male' error
print(tourismdf['Gender'].value_counts(normalize=True))
tdf['Gender'] = tdf['Gender'].replace({'Fe Male': 'Female'})
print(tdf['Gender'].value_counts(normalize=True))

# 7.2.Replace Single category with Unmarried category
print(tourismdf['MaritalStatus'].value_counts(normalize=True))
tdf['MaritalStatus'] = tdf['MaritalStatus'].replace({'Single': 'Unmarried'})
print(tdf['MaritalStatus'].value_counts(normalize=True))

# 8. Feature Engineering to Reduce Multicolinearity and Improve generalization
# 8.1. Create HasChildren Column based on NumberOfChildrenVisiting
tdf['HasChildren'] = (tdf['NumberOfChildrenVisiting'] > 0).astype(int)
print(tdf['HasChildren'].value_counts(normalize=True))


# 8.2. Create HasChildren Column AgeGroup based on Age
def AgeGroup(row):
    age = row['Age']
    if age <= 18:
        return 'Young'
    elif age >= 19 and age <= 40:
        return 'Adult'
    else:
        return 'Old'

tdf['AgeGroup'] = tdf.apply(AgeGroup, axis=1)
print(tdf['AgeGroup'].value_counts(normalize=True))


# 8.3. Monthly Income Categorization — Low, Mid, High
def IncomeCategory(row):
    income = row['MonthlyIncome']
    if income < 20000:
        return 'Low'
    elif income >= 20000 and income <= 30000:
        return 'Mid'
    else:
        return 'High'

tdf['IncomeCategory'] = tdf.apply(IncomeCategory, axis=1)
print(tdf['IncomeCategory'].value_counts(normalize=True))


# 8.4. Categorize DurationOfPitch — Short, Long, High
def PitchPeriodCategory(row):
    pitch = row['DurationOfPitch']
    if pitch <= 10:
        return 'Short'
    elif pitch >= 11 and pitch <= 30:
        return 'Long'
    else:
        return 'High'

tdf['PitchPeriodCategory'] = tdf.apply(PitchPeriodCategory, axis=1)
print(tdf['PitchPeriodCategory'].value_counts(normalize=True))


# 9. Set target column
target_col = "ProdTaken"

# 10. Split into X (features) and y (target)
X = tdf.drop(columns=[target_col])
y = tdf[target_col]

# 11. Perform train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

X_train.to_csv("X_train.csv",index=False)
X_test.to_csv("X_test.csv",index=False)
y_train.to_csv("y_train.csv",index=False)
y_test.to_csv("y_test.csv",index=False)


# 12. Upload train and test data files to Huggingface dataset
files = ["X_train.csv","X_test.csv","y_train.csv","y_test.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="supravab/Tourism_Package_Prediction",
        repo_type="dataset",
    )
