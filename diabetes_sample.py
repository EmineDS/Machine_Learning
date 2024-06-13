import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
data=pd.read_csv("./diabetes.csv")
pd.set_option("display.max_columns",None)
veri=data.copy()
print(veri)
y=veri.Outcome
X=veri.drop(columns="Outcome",axis=1)
X_train, X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
def modeller(model):
    model.fit(X_train,y_train)
    tahmin=model.predict(X_test)
    skor=accuracy_score(y_test,tahmin)
    return skor*100
models=[LogisticRegression(random_state=0),SVC(random_state=0),KNeighborsClassifier(),GaussianNB(),DecisionTreeClassifier(random_state=0)]

for i in models:
    print(i)
    print(modeller(i))

model=GaussianNB()
parameters = {
    'var_smoothing': np.logspace(0,-9, num=100)
}
grid=GridSearchCV(model,param_grid=parameters,cv=10)
grid.fit(X_train,y_train)
print(grid.best_params_)
#{'var_smoothing': 0.01}

model=GaussianNB(var_smoothing=0.23101297000831597)
model.fit(X_train,y_train)
tahmin=model.predict(X_test)
acs=accuracy_score(y_test,tahmin)
print(acs*100)