# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required library and read the dataframe 
2. Write a function computeCost to generate the cost function. 
3. Perform iterations og gradient steps with learning rate. 
4. Plot the cost function using Gradient Descent and generate the required graph.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: A Muhilan
RegisterNumber: 25015918
*/
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def linear_regression(X1, y, learning_rate=0.01, num_iters=1000):
    X = np.c_[np.ones(len(X1)), X1]
    theta = np.zeros(X.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
        predictions = (X).dot(theta).reshape(-1, 1)
        errors = (predictions - y).reshape(-1,1)
        theta -= learning_rate * (1 / len(X1)) * X.T.dot(errors)

    return theta
data = pd.read_csv('50_Startups.csv', header=None)
print(data.head())
X = (data.iloc[1:, :-2].values)
print(X)
X1 = X.astype(float)
scaler = StandardScaler()
y = (data.iloc[1:, -1].values).reshape(-1,1)
print(y)
X1_Scaled = scaler.fit_transform(X1)
y1_Scaled = scaler.fit_transform(y)
print('Name: Pragatheeshraaj D')
print('Register No.:212224230199')
print(X1_Scaled)
theta = linear_regression(X1_Scaled, y1_Scaled)
new_data = np.array([165349.2, 136897.8, 471784.1]).reshape(-1,1)
new_scaled = scaler.fit_transform(new_data)
prediction = np.dot(np.append(1, new_scaled), theta)
prediction = prediction.reshape(-1,1)
pre = scaler.inverse_transform(prediction)
print(f"Predicted value: {pre}")
```

## Output:

<img width="1496" height="450" alt="image" src="https://github.com/user-attachments/assets/0f7a4947-712b-45e7-9f9d-5dd5e2efe502" />
<img width="1493" height="442" alt="image" src="https://github.com/user-attachments/assets/f349e175-973d-44a8-abc4-956771834793" />
<img width="1492" height="421" alt="image" src="https://github.com/user-attachments/assets/83426509-bccf-43c3-80b9-1a7bd8523232" />
<img width="1491" height="442" alt="image" src="https://github.com/user-attachments/assets/28dbcd26-3da7-4d9f-b0f6-333a31418222" />
<img width="1489" height="442" alt="image" src="https://github.com/user-attachments/assets/9c375ddd-d702-4890-bb81-bb34f5380097" />
<img width="1491" height="452" alt="image" src="https://github.com/user-attachments/assets/cbe6da9a-b4f1-4b4c-a9c7-ff2de0fe14b5" />
<img width="1487" height="451" alt="image" src="https://github.com/user-attachments/assets/0300df54-9444-4eda-aaf2-2e644315d470" />
<img width="1493" height="415" alt="image" src="https://github.com/user-attachments/assets/d9fab409-8014-401d-9fd8-05c06adadfef" />
<img width="1484" height="267" alt="image" src="https://github.com/user-attachments/assets/4c1ae21c-ddec-46cd-b17a-6fef4f4be7d7" />



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
