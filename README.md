# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Yogaraj . S
RegisterNumber:  212223040248 
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df=pd.read_csv('/exp1.csv')
df.head(10)
plt.scatter(df['x'],df['y'])
plt.xlabel('x')
plt.ylabel('y')
x=df.iloc[:,0:1]
y=df.iloc[:,-1]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
x_train
y_train
lr.predict(x_test.iloc[0].values.reshape(1,1))
plt.scatter(df['x'],df['y'])
plt.xlabel('x')
plt.ylabel('y')
plt.plot(x_train,lr.predict(x_train),color='red')

```

## Output:
![simple linear regression model for predicting the marks scored](sam.png)

![Screenshot 2024-02-23 102313](https://github.com/yogaraj2/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/153482637/4407c1e5-a5b5-48af-a7aa-96dcb2a3aaf9)
![Screenshot 2024-02-23 102330](https://github.com/yogaraj2/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/153482637/89b72640-34aa-4abc-9ed6-41eb0e9d3e88)
![Screenshot 2024-02-23 102551](https://github.com/yogaraj2/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/153482637/e94a3c1c-3024-4103-8146-add0a0e70c0b)
![Screenshot 2024-02-23 102558](https://github.com/yogaraj2/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/153482637/59f9d898-f12c-4e78-a1fa-b8341e16ac59)
![Screenshot 2024-02-23 103600](https://github.com/yogaraj2/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/153482637/ee4b5567-a1b5-400c-b3ee-68839f8a75fa)




## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
