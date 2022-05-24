from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from IPython.display import Image 
import matplotlib.pyplot as plt
import pydotplus
import pandas as pd
import numpy as np

#load dataset
irisDataset = pd.read_csv("UTS\Data\iris.csv", sep=',', skiprows=0)
irisDataset.head()

#class target encoding
irisDataset["Species"] = pd.factorize(irisDataset.Species)[0]
# print(irisDataset)
print('-----------------\n')
print(f'Total Data : {len(irisDataset)}')

#menghapus kolom id
irisDataset = irisDataset.drop(labels="Id", axis=1)
# print(irisDataset)
print('-----------------\n')

#mengubah ke bentuk array numpy 
irisDataset = irisDataset.to_numpy()
# print(irisDataset)
print('-----------------\n')

# Mengambil 3 kolom pertama dari 5 kolom dan seluruh baris
X = irisDataset[:, 0:3]
print("Data Inputan: ")
print(X)
print('-----------------\n')
print(f'Total Data : {len(X)}')


# y = irisDataset[:, 4]
# print("\nData Label: ")
# print(y)
# print('-----------------\n')
