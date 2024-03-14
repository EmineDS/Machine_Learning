import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split,KFold
from sklearn.linear_model import LinearRegression
import sklearn.metrics as mt
from sklearn.preprocessing import PolynomialFeatures
def skor(model,x_train,x_test,y_train,y_test):
    egitimtahmin=model.predict(x_train)
    testtahmin=model.predict(x_test)
    #yüksek r 2 yi amaçlar
    r2_egitim=mt.r2_score(y_train,egitimtahmin)
    r2_test=mt.r2_score(y_test,testtahmin)
    #düşük hata değerleirni amaçlar
    mse_egitim=mt.mean_squared_error(y_train,egitimtahmin)
    mse_test=mt.mean_squared_error(y_test,testtahmin)

    rmse_egitim=mt.root_mean_squared_error(y_train,egitimtahmin)
    rmse_test = mt.root_mean_squared_error(y_test, testtahmin)

    mae_egitim=mt.mean_absolute_error(y_train,egitimtahmin)
    mae_test=mt.mean_absolute_error(y_test,testtahmin)
    return[r2_egitim,r2_test,mse_egitim,mse_test,rmse_egitim,rmse_test,mae_egitim,mae_test]
data=pd.read_csv("./ev.csv")
veri=data.copy()
print(veri.isnull().sum())
pd.set_option('display.max_columns', None)
print(veri)
outliers=["X2 house age","X3 distance to the nearest MRT station","X6 longitude","X5 latitude","Y house price of unit area"]

for i in outliers:
    Q1 = veri[i].quantile(0.25)
    Q3 = veri[i].quantile(0.75)
    IQR = Q3 - Q1
    ustsinir = Q3 + 1.5 * IQR
    altsinir = Q1 - 1.5 * IQR
    veri.loc[veri[i] < altsinir] = altsinir
    veri.loc[veri[i] > ustsinir] = ustsinir
for i in outliers:
    Q1 = veri[i].quantile(0.25)
    Q3 = veri[i].quantile(0.75)
    IQR = Q3 - Q1
    ustsinir = Q3 + 1.5 * IQR
    altsinir = Q1 - 1.5 * IQR
    veri.loc[veri[i] < altsinir] = altsinir
    veri.loc[veri[i] > ustsinir] = ustsinir
Y=veri["Y house price of unit area"]
X=veri.drop(columns=["No","X1 transaction date","Y house price of unit area"],axis=1)
pol=PolynomialFeatures(degree=2)
pol_X=pol.fit_transform(X)
lr_cv=LinearRegression()
k=6
iterasyon=1
cv=KFold(n_splits=k)
print(cv)
#K KATLI ÇAPRAZ DOĞRULAMA YAPALIM MODELİ ANLAMLANDIRMAK İÇİN
#CROSS VALIDATION
#en baştaki X ve Y değerlerimizi kullaıyoruz
for egitimindex,testindex in cv.split(pol_X):
    X_train,X_test=X.loc[egitimindex],X.loc[testindex]
    Y_train,Y_test=Y.loc[egitimindex],Y.loc[testindex]
    lr_cv.fit(X_train,Y_train)
    sonuc2=skor(model=lr_cv,x_train=X_train,x_test=X_test,y_train=Y_train,y_test=Y_test)
    print("İterasyon:{}".format(iterasyon))
    print("Eğitim R2={} Eğitim MSE={}".format(sonuc2[0], sonuc2[2]))
    print("Test R2={} Test MSE={}".format(sonuc2[1], sonuc2[3]))
    if iterasyon==3:
        mdl=lr_cv.predict(X_train)
        plt.plot(mdl, color="red")
        plt.plot(Y_test, color="blue")
        plt.show()

    iterasyon +=1
#K KATLI DOĞRULAMA YAPILMADAN Kİ SONUÇLAR İÇİN
X_train,X_test,Y_train,Y_test=train_test_split(pol_X,Y,test_size=0.2,random_state=42)
lr=LinearRegression()
lr.fit(X_train,Y_train)
tahmin=lr.predict(X_test)
r2skor=mt.r2_score(Y_test,tahmin)
mse=mt.mean_squared_error(Y_test,tahmin)
print(r2skor ,  mse)
Y_test=Y_test.sort_index()
plt.plot(tahmin, color="red")
plt.plot(Y_test, color="blue")
plt.show()