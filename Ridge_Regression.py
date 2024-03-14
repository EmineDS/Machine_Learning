import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Ridge
import sklearn.metrics as mt
import matplotlib.pyplot as plt
data=pd.read_csv(R"Linear_Regression/advertising.csv")
veri=data.copy()
print(veri)
y=veri["Sales"]
X=veri.drop(columns="Sales",axis=1)
print(X)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
lr=LinearRegression()
lr.fit(X_train,y_train)
tahmin=lr.predict(X_test)
r2=mt.r2_score(y_test,tahmin)
mse=mt.mean_squared_error(y_test,tahmin)
print("R2 DEĞERİ {}    MSE DEĞERİ  {}".format(r2,mse))
#eğer çoklu bağlantı probemi varsa r2 değeri olduğundan büyük de çıkabilir.
#ridge regresyonu bu sorunu çöer. çözmesiyle beraber r2 değeri de küçülecektir.
ridge_model=Ridge(alpha=0.1)#buradaki alpha parametresi bizim öğrendiğimiz lamda hiper parametresi değridir
ridge_model.fit(X_train,y_train)
tahmin2=ridge_model.predict(X_test)
r2rid=mt.r2_score(y_test,tahmin2)
mserid=mt.mean_squared_error(y_test,tahmin2)
print(
      "RİDGE REGRESYON İLE \n"
      "R2 DEĞERİ {}    MSE DEĞERİ  {}".format(r2rid,mserid))
#bizim çoklu doğrusal bağlantımız çok az olduğundan değişim de az oldu
#ridge model yapısında alpha değeri yükseldikçe  r2 değeri küçülür
katsayılar=[]
lambdalar=10**np.linspace(10,-2,100)*0.5
for i in lambdalar:
    ridmodel=Ridge(alpha=i)
    ridmodel.fit(X_train,y_train)
    katsayılar.append(ridmodel.coef_)#her alfa değeri için yeni model oluşturur
    # ve bu modellerin katsayı değerlerimi alır
ax=plt.gca()
ax.plot(lambdalar,katsayılar)
ax.set_xscale("log")
plt.xlabel("lambda")
plt.ylabel("katsayılar")
plt.show()
#burada lambda değerileri yükseldikçe katsayıların değerlerinin düştüğünü görebiliriz