import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import  sklearn.metrics as mt
from sklearn.linear_model import LinearRegression
data=pd.read_excel("./sicaklikverim.xlsx")
veri=data.copy()
print(veri)
Y=veri["Verim"]
X=veri["Sıcaklık"]
#eğitim ve test olarak parçalamadıysanız kendiniz array yapısı vermelisiniz
Y=Y.values.reshape(-1,1)
X=X.values.reshape(-1,1)
#dataframe türünden verileri array a çevirdik
lr=LinearRegression()
lr.fit(X,Y)#modelimizi data setinin tamamıyla eğittik
tahmin=lr.predict(X)#veri setinin X değerleriyle test ettik
r2dog=mt.r2_score(Y,tahmin)
mse=mt.mean_squared_error(Y,tahmin)
print(r2dog,mse)
#r2 değeri çok küçük çıktı
#elimizdeki x değişkenlerini değştir ve polinomal yaıya çevir dedik
pol=PolynomialFeatures(degree=3)
X_pol=pol.fit_transform(X)
lr2=LinearRegression()
lr2.fit(X_pol,Y)#POLİNOMSAL ŞEKİLDE EĞİTTİK
tahmin2=lr2.predict(X_pol)
r2pol=mt.r2_score(Y,tahmin2)
msepol=mt.mean_squared_error(Y,tahmin2)
print(r2pol,msepol)
#hem R2 büyüdük hem de hata göstergesi olan mse düştü
plt.scatter(X,Y,color="red")
plt.title("DOĞRUSAL MODEL DENEMESi")
plt.plot(X,tahmin,color="blue")
plt.show()
plt.scatter(X,Y,color="red")
plt.title("POLİNOMAL MODEL DENEMESİ")
plt.plot(X,tahmin2,color="blue")
plt.show()