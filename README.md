# EXNO:4-DS
# AIM:
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
![image](https://github.com/user-attachments/assets/efc79ab8-c61d-418a-a1af-dbfd594d8b43)
## data.isnull().sum()
![image](https://github.com/user-attachments/assets/ac3433ee-2d0c-4d30-ad0f-7025ffeefea3)
## missing=data[data.isnull().any(axis=1)]
![image](https://github.com/user-attachments/assets/5577f387-9107-4a9d-9119-fe758ffce303)
## data2=data.dropna(axis=0)
![image](https://github.com/user-attachments/assets/f47ecb24-7753-4e94-935b-0a4ce598d9a1)
```
sal=data['SalStat']
data2['SalStat']=data2['SalStat'].map({'less than or equal to 50,000':0,' greater than 50,000':1})
print(data['SalStat'])
```
![image](https://github.com/user-attachments/assets/fc752be0-cd1f-4375-9873-a69cfecf9554)
```
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![image](https://github.com/user-attachments/assets/658cff7d-83f9-4a7d-832c-61a17df46bb6)
## data2
![image](https://github.com/user-attachments/assets/07785ff8-e579-4e2c-aefe-8b1157c6b3d0)
## new_data=pd.get_dummies(data2, drop_first=True)
![image](https://github.com/user-attachments/assets/a2be719a-651c-4487-90aa-1165f0d7fe58)
## columns_list=list(new_data.columns)
![image](https://github.com/user-attachments/assets/599332c0-5e3c-4900-b52c-c60eecdecbe2)
## feature=list(set(columns_list)-set(['SalStat']))
![image](https://github.com/user-attachments/assets/5a6259cb-23ac-4c1b-9260-fe57a54404fd)
## y=new_data['SalStat'].values
![image](https://github.com/user-attachments/assets/d38f8527-6d79-45e8-9f41-74d34de45843)
## x=new_data[feature].values
![image](https://github.com/user-attachments/assets/b9c8cbf0-89bd-4415-b585-afa4b8231620)
```
KNN_classifier = KNeighborsClassifier(n_neighbors=5)
KNN_classifier.fit(train_x, train_y)
```
![image](https://github.com/user-attachments/assets/4b00db11-3b8e-4a33-ad90-d02a57e346c8)
## prediction=KNN_classifier.predict(test_x)
## confusionMatrix= confusion_matrix(test_y,prediction)
![image](https://github.com/user-attachments/assets/45ddc514-18b9-453a-a1fb-c8697cedbab3)
## accuracy_score=accuracy_score(test_y,prediction)
![image](https://github.com/user-attachments/assets/2e7b207b-b4d1-463d-9a09-ba9251d8a7a1)
## data.shape
![image](https://github.com/user-attachments/assets/319677d5-ec1c-40f3-922b-387bc9036c72)

# RESULT:
Thus feature scaling and selection is performed.
