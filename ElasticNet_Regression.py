import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge,Lasso,ElasticNet,ElasticNetCV
import sklearn.metrics as mt
df=fetch_california_housing()
data=pd.DataFrame(df.data,columns=df.feature_names)
veri=data.copy()
veri["PRICE"]=df.target
y=veri["PRICE"]
X=veri.drop(columns="PRICE",axis=1)
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.2)
#tüm modeller için eğitim işlemini gerçekleştirelim
ridge_model=Ridge(alpha=0.1)
ridge_model.fit(X_train,y_train)

lasso_model=Lasso(alpha=0.1)
lasso_model.fit(X_train,y_train)

elasticmodel=ElasticNet(alpha=0.1)
elasticmodel.fit(X_train,y_train)
#model başarısına bakalım
#print(ridge_model.score(X_train,y_train))
#print(lasso_model.score(X_train,y_train))
print(elasticmodel.score(X_train,y_train))

#print(ridge_model.score(X_test,y_test))
#print(lasso_model.score(X_test,y_test))
print(elasticmodel.score(X_test,y_test))

tahminrid=ridge_model.predict(X_test)
tahminlasso=lasso_model.predict(X_test)
tahminelastic=elasticmodel.predict(X_test)

#hata kareler yapısına da bakalım
#print(mt.mean_squared_error(y_test,tahminrid))
#print(mt.mean_squared_error(y_test,tahminlasso))
print(mt.mean_squared_error(y_test,tahminelastic))
#elastic model için cross validation yapalım
lamb=ElasticNetCV(cv=10,max_iter=10000).fit(X_train,y_train).alpha_
elasticmodel2=ElasticNet(alpha=lamb)
elasticmodel2.fit(X_train,y_train)
print(elasticmodel2.score(X_train,y_train))
print(elasticmodel2.score(X_test,y_test))
tahminelastic2=elasticmodel2.predict(X_test)
print(mt.mean_squared_error(y_test,tahminelastic2))

#cross validation ile score(r2) değeri yükseldi ve hata kareler değeri düştü