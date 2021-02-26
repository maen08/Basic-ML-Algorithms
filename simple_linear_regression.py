import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


'''
We need to predict results three(G3) based on first(G1) and second(G2) results of the students

'''

# import the whole data
data = pd.read_csv('student-mat.csv', sep=';')

# only the data needed, the rows you need
data = data[['G1', 'G2', 'G3', 'studytime', 'failures', 'absences']]

predict = 'G3'  # the value we need to predict

x = np.array(data.drop([predict], 1))  # whole data except G3
y = np.array(data[predict])

 #train_size, 10% of the data for test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1) 

linear = LinearRegression()
linear.fit(x_train, y_train)

model_accuracy = linear.score(x_test, y_test)
print(model_accuracy)


