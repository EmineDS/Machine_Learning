import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
data=pd.read_csv("./diabetes.csv")
veri=data.copy()
print(veri)
y=veri.Outcome
X=veri.drop(columns="Outcome",axis=1)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
model=RandomForestClassifier(random_state=0)
model.fit(X_train,y_train)
tahmin=model.predict(X_test)
acs=accuracy_score(y_test,tahmin)
print(acs*100)
parametreler={"n_estimators": range(100,500),"max_features":["sqrt", "log2", None],"max_depth":range(0,50),"max_leaf_nodes":range(0,100)}
grid=GridSearchCV(model,param_grid=parametreler,n_jobs=-1,cv=10)
grid.fit(X_train,y_train)
print(grid.best_params_)