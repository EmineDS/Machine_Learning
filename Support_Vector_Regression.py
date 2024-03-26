import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import matplotlib.pyplot as plt
data=pd.read_csv(R"./Position_Salaries.csv")
veri=data.copy()
print(veri)
y=veri["Salary"]
X=veri["Level"]
print(y)
#BU DATA SETİNİ EĞİTİM VE TEST YAPISINA AYIRMADIĞIMIZ İÇİN KENDİMİZ ARRAY YAPMAK ZORUNDAKALDIK
#NORMALDE TRAN SPİTİL FONKS ARRAY YAPISINA ÇEVİRİYODU
y=np.array(y).reshape(-1,1)
X=np.array(X).reshape(-1,1)
print(y)

scx=StandardScaler()
scy=StandardScaler()
y=scy.fit_transform(y)
X=scx.fit_transform(X)

svrmodel=SVR(kernel="rbf")#kersyonodelini tanımlar linear polynomsal rbf vb seçenekleri vrdır deneyerek en uygununu bul
svrmodel.fit(X,y)
tahmin=svrmodel.predict(X)

plt.scatter(X,y,color="red")
plt.plot(X,tahmin)
plt.show()