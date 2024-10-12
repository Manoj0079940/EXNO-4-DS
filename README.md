## AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
data=pd.read_csv('income.csv',na_values=[" ?"])
data
```
![image](https://github.com/user-attachments/assets/3192b6e7-1eb9-4ac1-85b4-f5417848e086)
## data.isnull().sum()
![image](https://github.com/user-attachments/assets/a2a99638-3674-4811-8b2c-bfe706208470)
## missing=data[data.isnull().any(axis=1)]
![image](https://github.com/user-attachments/assets/c934dd42-b2e3-4c62-b950-5f2758e72f18)
## data2=data.dropna(axis=0)
![image](https://github.com/user-attachments/assets/842c9617-1bf8-4951-af1e-8c32dad924c7)
```
sal=data['SalStat']
data2['SalStat']=data2['SalStat'].map({'less than or equal to 50,000':0,' greater than 50,000':1})
print(data['SalStat'])
```
![image](https://github.com/user-attachments/assets/5b23bbc9-16b7-48c4-ba8d-3da0e22b6cc8)
```
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![image](https://github.com/user-attachments/assets/ffc52e47-4e82-494e-85a6-a58d4ebd4181)
## data2
![image](https://github.com/user-attachments/assets/852962e9-ad3b-4eb8-b1ac-10410f365115)
## new_data=pd.get_dummies(data2, drop_first=True)
![image](https://github.com/user-attachments/assets/70ff8c0c-edcc-4cb3-8790-345f81e4ee2a)
## columns_list=list(new_data.columns)
[Uploading image.png…]()
## feature=list(set(columns_list)-set(['SalStat']))
![image](https://github.com/user-attachments/assets/45084b09-b2e3-42e8-a56f-05b20ccd3181)
## y=new_data['SalStat'].values
![image](https://github.com/user-attachments/assets/23a6b443-990a-4bfe-89e0-b4ddb9fc9421)
## x=new_data[feature].values
![image](https://github.com/user-attachments/assets/6240a7a7-58aa-4d41-bc66-6f9302eec6d2)
```
KNN_classifier = KNeighborsClassifier(n_neighbors=5)
KNN_classifier.fit(train_x, train_y)
```
![image](https://github.com/user-attachments/assets/7397ec7b-d9c5-4e60-b29c-2ab6a19c4a70)
## prediction=KNN_classifier.predict(test_x)
## confusionMatrix= confusion_matrix(test_y,prediction)
![image](https://github.com/user-attachments/assets/37a80b7b-3214-48d1-9a4c-d59e8756908d)
## accuracy_score=accuracy_score(test_y,prediction)
![image](https://github.com/user-attachments/assets/21a6b858-83a4-40e0-9114-649512e24123)
## data.shape
![image](https://github.com/user-attachments/assets/dc1ac30f-a6d1-4278-b83d-8984b1bdf534)

# RESULT:
             Thus feature scaling and selection is performed.
