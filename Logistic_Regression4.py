#sıralı logistic regresyon
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
data=pd.read_csv("./winequality-red.csv")
veri=data.copy()
pd.set_option("display.max_columns",None)
print(veri)
print(veri.isnull().sum())
print(veri.quality.unique())
kategori=["3","4","5","6","7","8"]
oe=OrdinalEncoder(categories=[kategori])
veri["Kalite"]=oe.fit_transform(veri["quality"].values.reshape(-1,1))
print(veri)
print(veri[veri["quality"]==3][["quality","Kalite"]])
print(veri[veri["quality"]==4][["quality","Kalite"]])
veri.drop(columns="quality",axis=1,inplace=True)
y=veri.Kalite
X=veri.drop(columns="Kalite",axis=1)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

model=LogisticRegression(random_state=0)
model.fit(X_train,y_train)
tahmin=model.predict(X_test)

cm=confusion_matrix(y_test,tahmin)
acs=accuracy_score(y_test,tahmin)
print(cm)
print(acs)