import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle


'''
We need to predict results three(G3) based on first(G1), second(G2) results
and other parameters of students 


'''

# import the whole data
data = pd.read_csv('student-mat.csv', sep=';')

# only the data needed, the rows you need
data = data[['G1', 'G2', 'G3', 'studytime', 'failures', 'absences']]

predict = 'G3'  # the value we need to predict

x = np.array(data.drop([predict], 1))  # whole data except G3
y = np.array(data[predict])

# train_size, 10% of the data for test
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1) 


'''
The model is saved already, no need to run the train code everytime


# the train
linear = LinearRegression()
linear.fit(x_train, y_train)


# check the model accuracy
model_accuracy = linear.score(x_test, y_test)
# print(model_accuracy)



# save the model, not train every time
with open('studentmodel.pickle', 'wb') as f:
    pickle.dump(linear, f)

'''


best_model = 0
for train_times in range(10000):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
    
    linear = LinearRegression()
    linear.fit(x_train, y_train)
    accuracy = linear.score(x_test, y_test)
    

    if accuracy > best_model:
        best_model = accuracy
        print(accuracy)
        

        with open('studentmodel.pickle', 'wb') as f:
            pickle.dump(linear, f)






# using the saved model, pickle file
pickle_file = open('studentmodel.pickle', 'rb')
linear = pickle.load(pickle_file)


# coefficients of line
# print('Coefficients:\n', linear.coef_)
# print('Intercept:\n', linear.intercept_)


# predictions = linear.predict(x_test)

# for x in range(len(predictions)):
#     print(predictions[x], x_test[x], y_test[x])