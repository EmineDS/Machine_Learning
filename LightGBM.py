from lightgbm import LGBMClassifier
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
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

model=LGBMClassifier()
model.fit(X_train,y_train)
tahmin=model.predict(X_test)
acs=accuracy_score(y_test,tahmin)
print(acs*100)
# parametreler={"learning_rate":[0.01,0.1],"n_estimators":[200,500,1000],"max_depth":[3,5,7],"subsample":[0.6,0.8,1.0]}
# grid=GridSearchCV(model,param_grid=parametreler,cv=10,n_jobs=-1)
# grid.fit(X_train,y_train)
# print(grid.best_params_)
#{'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 1000, 'subsample': 0.6}
model=LGBMClassifier(learning_rate=0.01,max_depth=3,n_estimators=1000,subsample=0.6)
model.fit(X_train,y_train)
tahmin=model.predict(X_test)
acs=accuracy_score(y_test,tahmin)
print(acs*100)