import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
from sklearn.linear_model import LinearRegression
import sklearn.metrics as mt
data=pd.read_csv("./winequality-red.csv")
veri=data.copy()
print(veri)
print(veri.isnull().sum())
print(veri.info())
#sns.pairplot(veri)
#plt.show()#bileşen sayısının fazladlığından
# görselleştirmede problem olduğunu görebiliyoruz
sns.heatmap(veri.corr(),annot=True,cbar=True)
plt.show()#buarada çoklu doğrusal bağlantı problemi olabileceğini de görmeliyiz
#bileşenleri ayıralım
y=veri["quality"]
X=veri.drop(columns="quality",axis=1)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
#pca için önce standartlaştırma işlemi yapamlıyız
#standart normal dağılıma çeviricez
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)
#normalleştirme işlemini sadece
#x değişkenlerin uygulamamız gerektiğini unutmayalım

pca=PCA()
#BU İŞLEMİ YAPARKEN BİLGİ KAYBINI minimum seviyede tutmak istenir

X_train2=pca.fit_transform(X_train)
X_test2=pca.fit_transform(X_test)
#x train ve x testi temel bileşen yapısına ayırdık

print(X_train.shape)
print(X_train2.shape)
#eğer bileşen sayısı düşmese bileşenin içine parametre girmeliyiz
pca2=PCA(n_components=1)
X_train2=pca2.fit_transform(X_train)
X_test2=pca2.fit_transform(X_test)
print(X_train.shape)
print(X_train2.shape)#bileşen sayısının 2 ye düştüğünü görebiliriz.
#n_component değerini hiper parametre olarak düşünebiliriz

print(np.cumsum(pca.explained_variance_ratio_)*100)#kümülatif toplam varyans aldık
#amaç açıklanan varyansın olabildiğince yüksek olması ve bileşen değerinin minimum olması amaçlanır
#açıklanan varyans için %90 ve üzeri kabul edilebilir değerdir.
plt.plot(np.cumsum(pca.explained_variance_ratio_))#kümülatif toplam varyans aldık
plt.xlabel("Bileşen Saysı")
plt.ylabel("Açıklanan Varyans")
plt.show()#çok düşük varyans çok fazla bilgi kaybı demek olabilir.
lm=LinearRegression()
lm.fit(X_train2,y_train)
tahmin=lm.predict(X_test2)

R2=mt.r2_score(y_test,tahmin)
rmse=mt.mean_squared_error(y_test,tahmin,squared=True)
print(R2,rmse)

