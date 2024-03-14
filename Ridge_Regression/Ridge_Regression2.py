import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Ridge,RidgeCV
import sklearn.metrics as mt
import numpy as np


data=pd.read_excel("./ornek.xlsx")
veri=data.copy()
print(veri)
#öncelikle bir korelasyon matirisi oluşturup değişkenler arasındaki
# korelasyona bakalım
sns.heatmap(veri.corr(),annot=True)
plt.show()
#yüksek korelasyonlar çoklu doğrusal bağlantı problemine
#sebep olur.
y=veri["Y"]
X=veri.drop(columns="Y",axis=1)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
lr=LinearRegression()
lr.fit(X_train,y_train)
tahmin=lr.predict(X_test)
r2=mt.r2_score(y_test,tahmin)
print(r2)#çoklu doğrusal bağlantı sorunu olduğu için r2 olması gerekenden yüksek çıktı
ridge=Ridge(alpha=0.1)
ridge.fit(X_train,y_train)
ridgetahmin=ridge.predict(X_test)
r2ridge=mt.r2_score(y_test,ridgetahmin)
print(r2ridge)
#lamda hiperparametremizin en optimal olanını seçmeliyiz
lambdalar=10**np.linspace(10,-2,100)*0.5#lambda değerleirni oluşturalım
#CROSS VALİDATİON YAPACAĞIZ ancak ridgenin kendi cross validation fonksiyonu vardır.
ridge_cv=RidgeCV(alphas=lambdalar,scoring="r2")
ridge_cv.fit(X_train,y_train)
print(ridge_cv.alpha_)#en uygun alpha değerini döndürür bu alpha değerini kullanalım.
ridge=Ridge(alpha=2018508.6292982749)
ridge.fit(X_train,y_train)
ridgetahmin=ridge.predict(X_test)
r2ridge=mt.r2_score(y_test,ridgetahmin)
print(r2ridge)#en optimal alpha değeriyle işlem yaptık ve çoklu doğrusal bağlantı problemi olduğundan
#alpha değerimizi bulduktan sonra r2 değerimiz gözle görülür şekilde düştü.