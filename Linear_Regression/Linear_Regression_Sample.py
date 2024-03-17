import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split, cross_val_score
import sklearn.metrics as mt
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
data=pd.read_csv("ev.csv")
print(data)
veri=data.copy()
print(data.info())
veri=veri.drop(columns="Address",axis=1)
print(veri.isnull().sum())
sns.pairplot(veri)
plt.show()
kor=veri.corr()
sns.heatmap(kor,annot=True)
plt.show()
y=veri["Price"]
X=veri.drop(columns="Price",axis=1)
constant=sm.add_constant(X)
vif=pd.DataFrame()
vif["Değişkenler"]=X.columns
vif["VİF"]=[variance_inflation_factor(constant,i+1) for i in range(X.shape[1])]
print(vif) #vif değerleri  ten küçükse çoklu bağlantısal problem yoktur
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

def caprazdog(model):
    dogruluk=cross_val_score(model,X,y,cv=10)#genelde k değeri 10 olarak kullanılır
    return dogruluk.mean()
def basari(gercek,tahmin):
    rmse=mt.mean_squared_error(gercek,tahmin,squared=True)
    r2=mt.r2_score(gercek,tahmin)
    return [rmse,r2]

lin_model=LinearRegression()
lin_model.fit(X_train,y_train)
lin_tahmin=lin_model.predict(X_test)

ridge_model=Ridge(alpha=0.1)
ridge_model.fit(X_train,y_train)
ridge_tahmin=ridge_model.predict(X_test)

lasso_model=Lasso(alpha=0.1)
lasso_model.fit(X_train,y_train)
lasso_tahmin=lasso_model.predict(X_test)

elac_model=ElasticNet(alpha=0.1)
elac_model.fit(X_train,y_train)
elac_tahmin=elac_model.predict(X_test)

sonuclar=[["Linear Model",basari(y_test,lin_tahmin)[0],basari(y_test,lin_tahmin)[1],caprazdog(lin_model)],
          ["Ridge Model",basari(y_test,ridge_tahmin)[0],basari(y_test,ridge_tahmin)[1],caprazdog(ridge_model)],
            ["Lasso Model",basari(y_test,lasso_tahmin)[0],basari(y_test,lasso_tahmin)[1],caprazdog(lasso_model)],
          ["ElasticNet Model",basari(y_test,elac_tahmin)[0],basari(y_test,elac_tahmin)[1],caprazdog(elac_model)]]
pd.options.display.float_format="{:.4f}".format
sonuclar=pd.DataFrame(sonuclar,columns=["Model","RMSE","R2","DOĞRULAMA"])
print(sonuclar)