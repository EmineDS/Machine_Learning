import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
data=pd.read_csv("./kanser.csv")
veri=data.copy()
print(veri)

M=veri[veri["diagnosis"]=="M"]
B=veri[veri["diagnosis"]=="B"]

plt.scatter(M.radius_mean,M.texture_mean,color="red",label="kötü huylu")
plt.scatter(B.radius_mean,B.texture_mean,color="green",label="iyi huylu")
plt.legend()
plt.show()

veri.drop(columns=["id","Unnamed: 32"],axis=1,inplace=True)
veri.diagnosis=[1 if kod=="M" else 0 for kod in veri.diagnosis]
print(veri)

y=veri.diagnosis
X=veri.drop(columns="diagnosis",axis=1)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

model=KNeighborsClassifier(n_neighbors=9)#9 değerini sonraki kodlarla belirledik
model.fit(X_train,y_train)
tahmin=model.predict(X_test)

acs=accuracy_score(y_test,tahmin)
print(acs*100)

basarı=[]
for k in range(1,20):
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    tahmin2=knn.predict(X_test)
    basarı.append(accuracy_score(y_test,tahmin2))

plt.plot(range(1,20),basarı)
plt.xlabel("K")
plt.ylabel("Başarı")
plt.show()

print(max(basarı))
