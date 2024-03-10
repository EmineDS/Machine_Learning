import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import sklearn.metrics as mt
data=pd.read_excel(".\\advertising2.xlsx")
veri=data.copy()
print(veri.isnull().sum())#eksik gözlme değerlerimiz var
#missing values için özel bir kütüphane var
imputer=SimpleImputer(missing_values=np.nan,strategy="mean")
imputer=imputer.fit(veri)
veri.iloc[:,:]=imputer.transform(veri)
print(veri.isnull().sum())
Y=veri["Sales"]
X=veri[["TV","Radio"]]
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20,random_state=42)
lr=LinearRegression()
lr.fit(X_train,Y_train)#eğitim işlemi
tahmin=lr.predict(X_test)
#model başarısına bakalım
r2=mt.r2_score(Y_test,tahmin)
mse=mt.mean_squared_error(Y_test,tahmin)
rmse=mt.root_mean_squared_error(Y_test,tahmin)
mae=mt.mean_absolute_error(Y_test,tahmin)
print(r2,mse,rmse,mae)
