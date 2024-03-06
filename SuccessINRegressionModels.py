import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error,mean_absolute_error
data=pd.read_excel(R".\Regressionsuccess.xlsx")
print(data)
Y=data["Y"]
X=data[["X1","X2"]]
sabit=sm.add_constant(X)
model=sm.OLS(Y,sabit).fit()
print(model.summary())
#ORTALAMA HATA KARE (MSE)(mean squared error)
#Hata değerimizin olması için önce tahmin modeli oluşturmamız gerekir
Tahmin=model.predict(sabit)#Y tahmin değerleri
print(Tahmin)
mse=mean_squared_error(Y,Tahmin)
print(mse)
#ORTALAMA HATA KARELER KÖKÜ (RMSE)(Root mean squared error)
rmse=mean_squared_error(Y,Tahmin,squared=False)#yalnızca squared parametresini false yapmamız yeterli
print(rmse)
#ORTALAMA MUTLAK HATALAR TOPLAMI(MAE)(Mean Absolute Error)
mae=mean_absolute_error(Y,Tahmin)
print(mae)