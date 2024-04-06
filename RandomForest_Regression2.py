import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor,RandomForestRegressor
import sklearn.metrics as mt
data=pd.read_csv("./Linear_Regression/advertising.csv")
veri=data.copy()
print(veri)
y=veri["Sales"]
X=veri.drop(columns="Sales",axis=1)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=42)



dtmodel=DecisionTreeRegressor(random_state=0)
dtmodel.fit(X_train,y_train)
dttahmin=dtmodel.predict(X_test)

bgmodel=BaggingRegressor(random_state=0)
bgmodel.fit(X_train,y_train)
bgtahmin=bgmodel.predict(X_test)

rfmodel=RandomForestRegressor(random_state=0)
rfmodel.fit(X_train,y_train)
rftahmin=rfmodel.predict(X_test)

r2dt=mt.r2_score(y_test,dttahmin)
r2bg=mt.r2_score(y_test,bgtahmin)
r2rf=mt.r2_score(y_test,rftahmin)

dtrmse=mt.mean_squared_error(y_test,dttahmin,squared=False)
bgrmse=mt.mean_squared_error(y_test,bgtahmin,squared=False)
rfrmse=mt.mean_squared_error(y_test,rftahmin,squared=False)

print("Karar Ağacı Regres.  R2 : {} RMSE: {}".format(r2dt,dtrmse))
print("Bagging Karar Ağacı  R2 : {} RMSE: {}".format(r2bg,bgrmse))
print("Random Forest Regre. R2 : {} RMSE: {}".format(r2rf,rfrmse))

#HER ÜÇ MODEL İÇİN DE TUNİNG YAPIP EN UYGUN PARAMETRELERİ BULALIM

# dtparametreler={"min_samples_split":range(2,20),"max_leaf_nodes":range(2,20)}
# dtgrid=GridSearchCV(estimator=dtmodel,param_grid=dtparametreler,cv=10,n_jobs=-1)
# dtgrid.fit(X_train,y_train)
# print(dtgrid.best_params_)

# sonuc
#{'max_leaf_nodes': 19, 'min_samples_split': 6}

# bgparametreler={"n_estimators":range(2,20)}
# bggrid=GridSearchCV(estimator=bgmodel,param_grid=bgparametreler,cv=10,n_jobs=-1)
# bggrid.fit(X_train,y_train)
# print(bggrid.best_params_)

#sonuc
#{'n_estimators': 13}

# rfparametreler={"max_depth":range(2,20),"max_features":range(2,20),"n_estimators":range(2,20)}
# rfgrid=GridSearchCV(estimator=rfmodel,param_grid=rfparametreler,cv=10,n_jobs=-1)
# rfgrid.fit(X_train,y_train)
# print(rfgrid.best_params_)

#sonuc
#{'max_depth': 13, 'max_features': 2, 'n_estimators': 19}

dtmodel=DecisionTreeRegressor(random_state=0,max_leaf_nodes=19, min_samples_split=6)
dtmodel.fit(X_train,y_train)
dttahmin=dtmodel.predict(X_test)

bgmodel=BaggingRegressor(random_state=0,n_estimators=13)
bgmodel.fit(X_train,y_train)
bgtahmin=bgmodel.predict(X_test)

rfmodel=RandomForestRegressor(random_state=0,max_depth=13, max_features=2, n_estimators=19)
rfmodel.fit(X_train,y_train)
rftahmin=rfmodel.predict(X_test)

r2dt=mt.r2_score(y_test,dttahmin)
r2bg=mt.r2_score(y_test,bgtahmin)
r2rf=mt.r2_score(y_test,rftahmin)

dtrmse=mt.mean_squared_error(y_test,dttahmin,squared=False)
bgrmse=mt.mean_squared_error(y_test,bgtahmin,squared=False)
rfrmse=mt.mean_squared_error(y_test,rftahmin,squared=False)
print(".............................model tuning sonrası....................")
print("Karar Ağacı Regres.  R2 : {} RMSE: {}".format(r2dt,dtrmse))
print("Bagging Karar Ağacı  R2 : {} RMSE: {}".format(r2bg,bgrmse))
print("Random Forest Regre. R2 : {} RMSE: {}".format(r2rf,rfrmse))

