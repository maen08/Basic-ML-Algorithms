import numpy as nm
import pandas as pd
import matplotlib.pyplot as plt

# import dataset

dataset = pd.read_csv('./salary.csv')

X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:, :-1].values