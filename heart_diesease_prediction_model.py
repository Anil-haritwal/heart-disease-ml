import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from  sklearn.metrics import accuracy_score

data=pd.read_csv(r"D:\New folder (5)\pracitce machine learning\1.py\heart_disease_data.csv")
# print(data.head())
# print(data.info())
# print(data.shape)
# print(data.describe())
# print(data.isnull().sum())
x=data.drop(columns="target",axis=1)
y=data['target']
print(x)
print(y)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,stratify=y,random_state=2)
model=LogisticRegression()
model.fit(x_train,y_train)
# accuracy on the train data
xprediction=model.predict(x_train)
acuracytrain=accuracy_score(xprediction,y_train)
print(acuracytrain)
testprediction=model.predict(x_test)
acuracytest=accuracy_score(testprediction,y_test)
print(acuracytest)
input=(44,1,1,120,263,0,1,173,0,0,2,0,3)

inputasnp=np.asarray(input)
reshaped=inputasnp.reshape(1,-1)
ans=model.predict(reshaped)
print(ans)