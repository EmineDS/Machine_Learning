import matplotlib.pyplot as plt
import pandas as pd
data=pd.read_csv("./Linear_Regression/advertising.csv")
from sklearn.model_selection import train_test_split,KFold
from sklearn.linear_model import LinearRegression
import sklearn.metrics as mt
veri=data.copy()
print(veri)
Y=veri["Sales"]
X=veri.drop(columns=["Sales"],axis=1)
print(X)
#eğitim ve test data setini ayırlarım

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20,random_state=42)
lr=LinearRegression()
lr.fit(X_train,Y_train)
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
sonuc1=skor(model=lr,x_train=X_train,x_test=X_test,y_train=Y_train,y_test=Y_test)
print("Eğitim R2={} Eğitim MSE={}".format(sonuc1[0],sonuc1[2]))
print("Test R2={} Test MSE={}".format(sonuc1[1],sonuc1[3]))

lr_cv=LinearRegression()
k=5
iterasyon=1
cv=KFold(n_splits=k)
print(cv)
#K KATLI ÇAPRAZ DOĞRULAMA
#CROSS VALIDATION
#en baştaki X ve Y değerlerimizi kullaıyoruz
for egitimindex,testindex in cv.split(X):
    X_train,X_test=X.loc[egitimindex],X.loc[testindex]
    Y_train,Y_test=Y.loc[egitimindex],Y.loc[testindex]
    lr_cv.fit(X_train,Y_train)
    sonuc2=skor(model=lr_cv,x_train=X_train,x_test=X_test,y_train=Y_train,y_test=Y_test)
    print("İterasyon:{}".format(iterasyon))
    print("Eğitim R2={} Eğitim MSE={}".format(sonuc2[0], sonuc2[2]))
    print("Test R2={} Test MSE={}".format(sonuc2[1], sonuc2[3]))
    iterasyon +=1
























