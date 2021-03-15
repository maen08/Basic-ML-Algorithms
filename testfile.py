import pickle
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


file_read = pd.read_csv('student-mat.csv', sep=';')

data = file_read[['G1', 'G2', 'G3', 'studytime']]
predict = 'G3'




x = np.array(data.drop([predict], 1))
y = np.array(data[predict])


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1) 

pickled = open('studentmodel.pickle', 'rb')
linear = pickle.load(pickled)

pred = linear.predict(x_test)

for i in range(len(pred)):

    print(pred[i], y_test[i])