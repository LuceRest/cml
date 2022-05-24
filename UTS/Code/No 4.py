#Importing library yang akan digunakan
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
print(irisDataset)
print('-----------------\n')

#menghapus kolom id
irisDataset = irisDataset.drop(labels="Id", axis=1)
print(irisDataset)
print('-----------------\n')

#mengubah ke bentuk array numpy 
irisDataset = irisDataset.to_numpy()
print(irisDataset)
print('-----------------\n')

#memisahkan input dengan label
x = irisDataset[:, 0:4]
y = irisDataset[:, 4]
print("Data Inputan: ")
print(x)
print('-----------------\n')

print("\nData Label: ")
print(y)
print('-----------------\n')

#spliting data training dan data testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
print("Data Training: ")
print(x_train)
print(len(x_train))
print('-----------------\n')

print("Label Data Training: ")
print(y_train)
print(len(y_train))
print('-----------------\n')

print("Data Testing: ")
print(x_test)
print(len(x_test))
print('-----------------\n')

print("Label Data Testing: ")
print(y_test)
print(len(y_test))
print('-----------------\n')


#menyiapkan model 
decisiontree = DecisionTreeClassifier(criterion= "entropy",
                                      random_state=0, max_depth=5,
                                      min_samples_split=2, min_samples_leaf=50, 
                                      min_weight_fraction_leaf=0, max_leaf_nodes=100, 
                                      min_impurity_decrease=0)
print("Model Siap Digunakan!")

#Training Model 
model = decisiontree.fit(x_train, y_train)
print("Proses Training Selesai!")
print('-----------------\n')

#Testing Model
y_pred = model.predict(x_test)
probabilitas = model.predict_proba(x_test)
print("Label Sebenarnya: ")
print(y_test)
print('-----------------\n')

print("Label Prediksi: ")
print(y_pred)
print('-----------------\n')

print("Nilai Confidence: ")
print(probabilitas)
print('-----------------\n')

