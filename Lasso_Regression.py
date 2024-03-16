import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso,LassoCV
import sklearn.metrics as mt

housing = fetch_california_housing()
data=pd.DataFrame(housing.data,columns=housing.feature_names)
print(data)
veri=data.copy()
veri["PRICE"]=housing.target
print(veri)
y=veri["PRICE"]
X=veri.drop(columns="PRICE",axis=1)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
ridge_model=Ridge(alpha=0.1)
ridge_model.fit(X_train,y_train)
print(ridge_model.score(X_train,y_train))
print(ridge_model.score(X_test,y_test))
tahmin=ridge_model.predict(X_test)
print(mt.r2_score(y_test,tahmin))
lasso_model=Lasso(alpha=0.01)#amacımız alpha değerini olabildiğince en küçük değerde kullanmak
lasso_model.fit(X_train,y_train)
print(lasso_model.score(X_train,y_train))
print(lasso_model.score(X_test,y_test))
print(ridge_model.coef_)
print(lasso_model.coef_)
#CROSS VALİDATİON YAPALIM
#bunu lambda tahmini için yapıyoruz.
lamb=LassoCV(cv=10,max_iter=10000).fit(X_train,y_train).alpha_#buradaki cv değeri k hiper parametresidir.
print(lamb)#en uygun lambda değerini buldu

lasso_model=Lasso(alpha=lamb)
lasso_model.fit(X_train,y_train)
print(lasso_model.score(X_train,y_train))
print(lasso_model.score(X_test,y_test))
#r2 değerlerinin yükseldiğini göreviliriz
