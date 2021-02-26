import numpy as nm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# import dataset

dataset = pd.read_csv('./salary.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,:-1].values


# separate data into test and train
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)
