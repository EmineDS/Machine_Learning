import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import sklearn.metrics as mt
from sklearn.ensemble import BaggingRegressor
data=pd.read_csv(R"./Linear_Regression/advertising.csv")
veri=data.copy()
print(veri)

y=veri["Sales"]
X=veri.drop(columns="Sales",axis=1)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
#ÖNCE KARAR AĞACI REGRESYONUNA BAKALIM
dtmodel=DecisionTreeRegressor(random_state=0)
dtmodel.fit(X_train,y_train)
dttahmin=dtmodel.predict(X_test)
r2=mt.r2_score(y_test,dttahmin)
rmse=mt.mean_squared_error(y_test,dttahmin,squared=False)
print("KARA AĞACI REGRESYON R2 {}   RMSE {}".format(r2,rmse))

# BAGGING ILE OLAN FARKA BAKALIM

bgmodel=BaggingRegressor(random_state=0)
bgmodel.fit(X_train,y_train)
bgtahmin=bgmodel.predict(X_test)

r2_2=mt.r2_score(y_test,bgtahmin)
rmse_2=mt.mean_squared_error(y_test,bgtahmin,squared=False)
print("BAGGING REGRESYON R2 {}   RMSE {}".format(r2_2,rmse_2))
