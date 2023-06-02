#!/usr/bin/env python
# coding: utf-8

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics as m

s_data = pd.read_csv("https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv")

print("data successfully imported")
s_data.head(5)

s_data.plot(x="Hours",y="Scores",style="*")
plt.title("Hours vs Percentage")
plt.xlabel("Hours studied")
plt.ylabel("Percentage scored")
plt.show()

x = s_data.iloc[:,:-1].values
y = s_data.iloc[:,1].values

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)
regressor = LinearRegression()
regressor.fit(x_train.reshape(-1,1),y_train)
print("Training Complete")

line=regressor.coef_*x + regressor.intercept_
plt.scatter(x,y)
plt.plot(x,line,color="blue")
plt.show()

print(x_test)
y_pred = regressor.predict(x_test)

df = pd.DataFrame({"actual":y_test,"predicted":y_pred})
df

print("Training Score:",regressor.score(x_train,y_train))
print("Test Score:", regressor.score(x_test,y_test))

df.plot(kind="kde")
plt.show

hours = 9.25
test = np.array([hours])
test = test.reshape(-1,1)
own_pred = regressor.predict(test)
print("no. of hrs ={}".format(hours))
print("predicted Scores ={}".format(own_pred[0]))

print("mean abs error:",m.mean_absolute_error(y_test, y_pred))
print("mean squared error:",m.mean_squared_error(y_test, y_pred))
print("root mean squared error:",np.sqrt(m.mean_squared_error(y_test, y_pred)))
print("R-2:",m.r2_score(y_test, y_pred))

