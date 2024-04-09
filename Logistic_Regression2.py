import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report,roc_curve,roc_auc_score
import matplotlib.pyplot as plt
data=pd.read_csv(R"./kanser.csv")
veri=data.copy()
veri.drop(columns=["id","Unnamed: 32"],axis=1,inplace=True)
print(veri)
#bu veri setinde M kötü huylu B iyi huylu tümördür

veri.diagnosis=[1 if kod=="M" else 0 for kod in veri.diagnosis]
# M ve B değerlerini 1 ve 0 olarak kodladık
print(veri)

y=veri["diagnosis"]
X=veri.drop(columns="diagnosis",axis=1)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

model=LogisticRegression(random_state=0)
model.fit(X_train,y_train)
tahmin=model.predict(X_test)

cm=confusion_matrix(y_test,tahmin)
print(cm)

acs=accuracy_score(y_test,tahmin)
print(acs)

cr=classification_report(y_test,tahmin)
print(cr)

auc=roc_auc_score(y_test,tahmin)
fpr,tpr,theresold=roc_curve(y_test,model.predict_proba(X_test)[:,1])
plt.plot(fpr,tpr,label="Model Auc(Alan=%0.2f)"%auc)
plt.plot([0,1],[0,1],"r--")
plt.xlabel("False Positive Ratio")
plt.ylabel("True Positive Ratio")
plt.legend()
plt.title("ROC CURVE")

plt.show()