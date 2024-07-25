# PRODIGY_DS_02
import os
import kaggle
import pandas as pd

# Ensure the .kaggle directory exists
kaggle_dir = r'C:\Users\LENOVO\.kaggle'
os.makedirs(kaggle_dir, exist_ok=True)

# Define the path for the kaggle.json file
kaggle_json_path = os.path.join(kaggle_dir, 'kaggle.json')

# Check if kaggle.json exists
if not os.path.isfile(kaggle_json_path):
    raise FileNotFoundError(f'{kaggle_json_path} not found. Please place your kaggle.json file in the .kaggle directory.')

# Initialize the Kaggle API with authentication
kaggle.api.authenticate()

# Download the datasets
try:
    kaggle.api.dataset_download_files('c/titanic', path='data', unzip=True)
except Exception as e:
    print(f'Error: {e}')

# Read the CSV files into DataFrames
try:
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    gender_submission_df = pd.read_csv('data/gender_submission.csv')

    # Display the first few rows of each DataFrame
    print("Train DataFrame:")
    print(train_df.head())  # Use the correct variable name

    print("\nTest DataFrame:")
    print(test_df.head())

    print("\nGender Submission DataFrame:")
    print(gender_submission_df.head())
except FileNotFoundError as e:
    print(f'File not found: {e}')
except pd.errors.EmptyDataError as e:
    print(f'Error reading CSV file: {e}')







# Display the first few rows and check column names
print(train_df.head())
print(train_df.columns)

# Check for missing columns or data issues
if 'Embarked' in train_df.columns:
    print(train_df['Embarked'].isnull().sum(), "missing values in 'Embarked'")

# Data Cleaning for Train Dataset
if 'Age' in train_df.columns:
    train_df['Age'].fillna(train_df['Age'].median(), inplace=True)
else:
    print("Column 'Age' not found in train dataset.")

if 'Embarked' in train_df.columns:
    train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace=True)
else:
    print("Column 'Embarked' not found in train dataset.")

if 'Cabin' in train_df.columns:
    train_df.drop(columns=['Cabin'], inplace=True)
else:
    print("Column 'Cabin' not found in train dataset.")

# Data Cleaning for Test Dataset
if 'Age' in test_df.columns:
    test_df['Age'].fillna(test_df['Age'].median(), inplace=True)
else:
    print("Column 'Age' not found in test dataset.")

if 'Fare' in test_df.columns:
    test_df['Fare'].fillna(test_df['Fare'].median(), inplace=True)
else:
    print("Column 'Fare' not found in test dataset.")

if 'Cabin' in test_df.columns:
    test_df.drop(columns=['Cabin'], inplace=True)
else:
    print("Column 'Cabin' not found in test dataset.")

# Summary statistics of the training dataset
print(train_df.describe())

import matplotlib.pyplot as plt
import seaborn as sns

# Set the style of the plots
sns.set(style="whitegrid")

# Count plot of survivors
plt.figure(figsize=(10, 6))
sns.countplot(x='Survived', data=train_df)
plt.title('Count of Survivors')
plt.show()

# Count plot of passengers by class
plt.figure(figsize=(10, 6))
sns.countplot(x='Pclass', data=train_df)
plt.title('Count of Passengers by Class')
plt.show()

# Distribution of ages
plt.figure(figsize=(10, 6))
sns.histplot(train_df['Age'], kde=True)
plt.title('Distribution of Ages')
plt.show()

# Box plot of ages by class
plt.figure(figsize=(10, 6))
sns.boxplot(x='Pclass', y='Age', data=train_df)
plt.title('Box Plot of Ages by Class')
plt.show()

# Count plot of embarked locations
plt.figure(figsize=(10, 6))
sns.countplot(x='Embarked', data=train_df)
plt.title('Count of Embarked Locations')
plt.show()

# Survival rate by class
plt.figure(figsize=(10, 6))
sns.barplot(x='Pclass', y='Survived', data=train_df)
plt.title('Survival Rate by Class')
plt.show()

# Survival rate by sex
plt.figure(figsize=(10, 6))
sns.barplot(x='Sex', y='Survived', data=train_df)
plt.title('Survival Rate by Sex')
plt.show()

# Survival rate by age
plt.figure(figsize=(10, 6))
sns.histplot(train_df[train_df['Survived'] == 1]['Age'], kde=True, color='green', label='Survived')
sns.histplot(train_df[train_df['Survived'] == 0]['Age'], kde=True, color='red', label='Did not survive')
plt.title('Survival Rate by Age')
plt.legend()
plt.show()

# Pairplot to see relationships between features
sns.pairplot(train_df.dropna(), hue='Survived', diag_kind='kde')
plt.show()

# Merging the gender submission with the test dataset for comparison
test_df_with_submission = test_df.merge(gender_submission_df, on='PassengerId')
print(test_df_with_submission.head())

