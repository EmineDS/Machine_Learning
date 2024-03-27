import numpy
import yfinance as yf
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import sklearn.metrics as mt
from sklearn.model_selection import GridSearchCV
data=yf.download("THYAO.IS",start="2022-08-01",end="2022-09-01")#başlangıçı kapsar sonu kapsamaz
veri=data.copy()
veri=veri.reset_index()
print(veri["Date"].astype(str).str.split("-").str[2])
veri["Day"]=veri["Date"].astype(str).str.split("-").str[2]
print(veri)

y=veri["Adj Close"]
X=veri["Day"]
X=np.array(X).reshape(-1,1)
y=np.array(y).reshape(-1,1)
SCX=StandardScaler()
SCy=StandardScaler()
X=SCX.fit_transform(X)
y=SCy.fit_transform(y)
svrmodel=SVR(kernel="rbf",C=10000) #keernel default olarak rbf gelir
#C değeri de bir hiper parametredir. cezalandırmayla ilgilidir.
#kernel da bir hiperparametredir
svrmodel.fit(X,y)
tahminrbf=svrmodel.predict(X)
svrmodel=SVR(kernel="poly",degree=5)
#burada degree de bir hiperparametredir.default olarak 3 verilir
svrmodel.fit(X,y)
tahminpoly=svrmodel.predict(X)
svrmodel=SVR(kernel="linear")
svrmodel.fit(X,y)
tahminlinear=svrmodel.predict(X)
plt.scatter(X,y,color="red")
plt.plot(X,tahminrbf,color="blue",label="rbf")
plt.plot(X,tahminpoly,color="green",label="poly")
plt.plot(X,tahminlinear,color="black",label="linear")
plt.grid()
plt.legend()
plt.show()
r2=mt.r2_score(X,tahminrbf)

rmse=mt.mean_squared_error(X,tahminrbf,squared=False)
print(r2,
      rmse)

parametreler={"C":[1,10,100,1000,10000],"gamma":[1,0.1,0.001],"kernel":["rbf","linear","poly"]}
#parametreler için hangi değerleri deneyeceğimizi yazdık
tuning=GridSearchCV(estimator=SVR(),param_grid=parametreler,cv=10)
tuning.fit(X,y)
print(tuning.best_params_)
#belirlenen en optimal parametrelere göre modeli yeniden yazalım
svrmodel=SVR(kernel="rbf",C=100,gamma=1)
svrmodel.fit(X,y)
tahminrbf=svrmodel.predict(X)
r2=mt.r2_score(X,tahminrbf)

rmse=mt.mean_squared_error(X,tahminrbf,squared=False)
print(r2,
      rmse)
plt.scatter(X,y,color="red")
plt.plot(X,tahminrbf,color="blue",label="rbf")
plt.legend()
plt.show()