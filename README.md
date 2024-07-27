import subprocess
import os
import zipfile
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

os.environ['KAGGLE_USERNAME'] = '**********' #Put your user name from the API Token downloaded from your kaggle account
os.environ['KAGGLE_KEY'] = ' *********************' #Put your key from the API Token downloaded from your kaggle account

kaggle_path = r'C:\Users\LENOVO\AppData\Roaming\Python\Python312\Scripts\kaggle.exe'
download_path = 'C:/Users/LENOVO/Downloads/titanic_data'
os.makedirs(download_path, exist_ok=True)
subprocess.run([kaggle_path, 'competitions', 'download', '-c', 'titanic', '-p', download_path])

zip_file_path = os.path.join(download_path, 'titanic.zip')  # Ensure this matches the downloaded ZIP file name
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(download_path)

train_data = pd.read_csv(os.path.join(download_path, 'train.csv'))
test_data = pd.read_csv(os.path.join(download_path, 'test.csv'))
gender_submission = pd.read_csv(os.path.join(download_path, 'gender_submission.csv'))

print(train_data.isnull().sum())
train_data['Age'] = train_data['Age'].fillna(train_data['Age'].median())
train_data['Embarked'] = train_data['Embarked'].fillna(train_data['Embarked'].mode()[0])
train_data.drop(columns=['Cabin'], inplace=True)
train_data['Sex'] = train_data['Sex'].map({'male': 0, 'female': 1})
train_data = pd.get_dummies(train_data, columns=['Embarked'], drop_first=True)
numeric_train_data = train_data.select_dtypes(include=[np.number])

#correlation matrix
corr = numeric_train_data.corr()

# Visualization
def add_labels(ax, labels, is_percentage=False):
    for p in ax.patches:
        height = p.get_height()
        text = f'{height:.2f}%' if is_percentage else f'{height}'
        ax.annotate(text, (p.get_x() + p.get_width() / 2., height),
        ha='center', va='bottom', fontsize=12)





# Survival counts
plt.figure(figsize=(8, 6))
ax = sns.countplot(x='Survived', data=train_data)
plt.title('Survival Counts')
add_labels(ax, train_data['Survived'].value_counts())
plt.xlabel('Survived (0: No, 1: Yes)')
plt.show()

# Survival Rate by Gender
plt.figure(figsize=(8, 6))
ax = sns.barplot(x='Sex', y='Survived', data=train_data)
plt.title('Survival Rate by Gender')

# Calculate survival rate percentages by gender
survival_rates = train_data.groupby('Sex')['Survived'].mean() * 100
add_labels(ax, survival_rates, is_percentage=True)
plt.xlabel('Gender (0: Male, 1: Female)')
plt.ylabel('Survival Rate (%)')
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()



