# yg paling penting library sklearn dalam machine learning.
# sklearn = menyediakan tools" untuk machine learning.

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
irisDataset = pd.read_csv("6th Meet (Wednesday, March, 16th 2022)\Practice\Data\iris.csv", sep=',', skiprows=0)
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
                                      random_state=0, max_depth=10,
                                      min_samples_split=2, min_samples_leaf=1, 
                                      min_weight_fraction_leaf=0, max_leaf_nodes=None, 
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


#nilai akurasi testing
prediksiBenar = (y_pred == y_test).sum()
prediksiSalah = (y_pred != y_test).sum()
print("Prediksi Benar: ", prediksiBenar, "data")
print('-----------------\n')

print("Prediksi Salah: ", prediksiSalah, "data")
print('-----------------\n')

akurasi = prediksiBenar/(prediksiBenar+prediksiSalah)
print("Akurasi Model: ", akurasi)
print('-----------------\n')


#confusion matrix
pd.DataFrame(
confusion_matrix(y_test, y_pred),
index=['True:Iris-setosa','True:Iris-versicolor', 'True:Iris-virginica'],
columns=['Pred:Iris-sentosa','Pred:Iris-versicolor','Pred:Iris-virginica'],
)

#membuat visualisasi decision tree
nama_feature = np.array([["Sepal Length (cm)"], ["Sepal Width (cm)"], ["Petal Length (cm)"], ["Petal Width (cm)"]])

nama_kelas = np.array(["Iris-setosa", "Iris-versicolor", "Iris-virginica"])

dot_data = tree.export_graphviz(decisiontree, out_file=None, feature_names = nama_feature, class_names = nama_kelas)
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())
graph.write_png("6th Meet (Wednesday, March, 16th 2022)\Practice\Image\IrisDecisionTree-1.png")
print("Grafik Decision Tree Telah Di Export!")

fig, axes = plt.subplots(nrows = 1, ncols  = 1, figsize = (4,4), dpi = 300)
tree.plot_tree(decisiontree,
               feature_names=nama_feature,
               class_names=nama_kelas,
               filled=True)
fig.savefig("6th Meet (Wednesday, March, 16th 2022)\Practice\Image\IrisDecisionTree-2.png")

Image(filename="6th Meet (Wednesday, March, 16th 2022)\Practice\Image\IrisDecisionTree-1.png")
Image(filename="6th Meet (Wednesday, March, 16th 2022)\Practice\Image\IrisDecisionTree-2.png")

'''
note:
data versicolor,setosa, dan virginca dibuat menjadi angka supaya tipe data nya sama sehingga memudahkan proses.
axis grafik terdapat 2 bentuk, axis 1 untuk menghapus kolom, axis 0 untuk menghapus baris.
tidak ada pemangkasan dalam proses karena menggunakan library.
dihitung rasio yg sesuai di masing" class setiap label di dalam random state.
jenis random state: 42/1234
max dept sma dngn level jika.
'''






