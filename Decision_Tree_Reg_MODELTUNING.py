import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.tree import DecisionTreeRegressor
import sklearn.metrics as mt
data=pd.read_csv(R"./IceCreamData.csv")
veri=data.copy()
print(veri)#sıcaklık ve buna bağl olarak satışlar veril
y=veri["Revenue"]
X=veri["Temperature"]
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.1)
dtr=DecisionTreeRegressor(random_state=42)
dtr.fit(X_train.values.reshape(-1,1),y_train)
tahmin=dtr.predict(X_test.values.reshape(-1,1))
print(tahmin)

r2=mt.r2_score(y_test,tahmin)
mse=mt.mean_squared_error(y_test,tahmin)
print("R2 {}  MSE {}".format(r2,mse))

parametreler={"min_samples_split":range(2,50),
              "max_leaf_nodes":range(2,50)}

grid=GridSearchCV(estimator=dtr,param_grid=parametreler,cv=10)
grid.fit(X_train.values.reshape(-1,1),y_train)
print(grid.best_params_)
#bulduğumuz optimal değerler ile tekrar modelleme yapıp model başarısına bakalım

dtr=DecisionTreeRegressor(random_state=42,max_leaf_nodes=21,min_samples_split=17)
dtr.fit(X_train.values.reshape(-1,1),y_train)
tahmin=dtr.predict(X_test.values.reshape(-1,1))
print(tahmin)

r2=mt.r2_score(y_test,tahmin)
mse=mt.mean_squared_error(y_test,tahmin)
print("R2 {}  MSE {}".format(r2,mse))