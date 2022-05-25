from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

#Load Dataset
irisDataset = pd.read_csv("8th Meet (Wednesday, April, 13rd 2022)\Practice\Data\iris.csv", sep=',', skiprows=0)
irisDataset.head()

#Visualisasi Data
sns.set(style="white", color_codes=True)
sns.FacetGrid(irisDataset, hue="Species", height=6).map(plt.scatter, "Petal Length (cm)", "Petal Width (cm)").add_legend()


# Memisahkan atribut dengan target class 
x = irisDataset.iloc[:, 1:5].values
y = irisDataset["Species"].values
print("Atribut Dataset :")
print(x)
print('-----------------\n')
print("Class Target Dataset :")
print(y)
print('-----------------\n')


# Spliting data training dan data testing
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


#Normalisasi dataset 
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)
print("Data Training Setelah Normalisasi Dataset :")
print(X_train)
print('-----------------\n')
print("Data Testing Setelah Normalisasi Dataset :")
print(X_test)
print('-----------------\n')


#Training NBC Model 
model = GaussianNB()
model.fit(X_train, y_train)
print("Proses TraiKNN.ipynbning Selesai!")
print('-----------------\n')


#Melakukan testing terhadap model NBC
y_pred = model.predict(X_test)
print("Hasil Prediksi Data Testing Menggunakan NBC :")
print(y_pred)
print('-----------------\n')


#Melakukan perbandingan hasil prediksi label dengan data label asli
compare = pd.DataFrame({'Real Values':y_test, 'Predicted Values':y_pred})
print("Perbandingan Hasil Prediksi :")
print(compare)
print('-----------------\n')


#Hasil akurasi testing
prediksiBenar = (y_pred == y_test).sum()
prediksiSalah = (y_pred != y_test).sum()
print("Prediksi Benar : ", prediksiBenar, "data")
print('-----------------\n')
print("Prediksi Salah : ", prediksiSalah, "data")
print('-----------------\n')
akurasi = prediksiBenar/(prediksiBenar+prediksiSalah)
print("Akurasi Model : ", akurasi)
print('-----------------\n')



#condfusion matrix
pd.DataFrame(
    confusion_matrix(y_test, y_pred),
    index=['True : Iris-setosa', 'True : Iris-versicolor', 'True : Iris-virginica'],
    columns=['Pred : Iris-setosa', 'Pred : Iris-versicolor', 'Pred : Iris-virginica'],
)


#Contoh memprediksi 1 data
data = np.zeros((1, 4), dtype=float)
for x in range(4):
    if x == 0:
        print("Masukan nilai sepal length (cm) : ")
        a = float(input())
        data[0][x] = a
    elif x == 1:
        print("Masukan nilai sepal width (cm) : ")
        a = float(input())
        data[0][x] = a
    elif x == 2:
        print("Masukan nilai sepal length (cm) : ")
        a = float(input())
        data[0][x] = a
    elif x == 3:
        print("Masukan nilai sepal length (cm) : ")
        a = float(input())
        data[0][x] = a
print("Data yang ingin diprediksi : ", data)
print('-----------------\n')
data = sc.transform(data)
print("Hasil Normalisasi Data : ", data)
print('-----------------\n')

#Klasifikasi data
prediction = model.predict(data)
print("Hasil Prediksi : ", prediction)
print('-----------------\n')



