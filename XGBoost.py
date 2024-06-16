from xgboost import XGBClassifier
import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
data=pd.read_csv("./diabetes.csv")
veri=data.copy()
print(veri)
y=veri.Outcome
X=veri.drop(columns="Outcome",axis=1)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
xgb=XGBClassifier()
xgb.fit(X_train,y_train)
tahmin=xgb.predict(X_test)
acs=accuracy_score(y_test,tahmin)
print(acs*100)
# parametreler={"max_depth":[3,5,7],"subsample":[0.2,0.5,0.7],"n_estimators":[500,1000,2000],"learning_rate":[0.2,0.5,0.7]}
# grid=GridSearchCV(xgb,param_grid=parametreler,n_jobs=-1,cv=10)
# grid.fit(X_train,y_train)
# print(grid.best_params_)
#{'learning_rate': 0.2, 'max_depth': 7, 'n_estimators': 500, 'subsample': 0.7}
xgb=XGBClassifier(learning_rate=0.2, max_depth=7, n_estimators=500, subsample=0.7)
xgb.fit(X_train,y_train)
tahmin=xgb.predict(X_test)
acs=accuracy_score(y_test,tahmin)
print(acs*100)