import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import sklearn.metrics as mt
data=sns.load_dataset("tips")
veri=data.copy()
pd.set_option('display.max_columns', None)
print(veri)
print(veri.isnull().sum())
print(veri.dtypes)
#kategorik değişkenlere kodlama işlemi yapmalıyız.
kategori=[]
kategorik=veri.select_dtypes(include=["category"])#data timi kategori olanları seçip aldık
print(kategorik)
#kategorik değişkenlerin başlık değerlerini ayıralım
for i in kategorik.columns:
    kategori.append(i)
print(kategori)
#kukla değişken tuzağına düşmemek için drop first parametresini kullandık
veri=pd.get_dummies(veri,columns=kategori,dtype=int,drop_first=True)
print(veri)
Y=veri["tip"]
X=veri.drop(columns="tip",axis=1)
print(X)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20,random_state=10)
lr=LinearRegression()
lr.fit(X_train,Y_train)#eğitim yaptık
tahmin=lr.predict(X_test)
Y_test=Y_test.sort_index()
df=pd.DataFrame({"Gerçek":Y_test,"Tahmin":tahmin})
df.plot(kind="line")
plt.show()
print(mt.r2_score(Y_test,tahmin))
#r2 değeri - çıktı yani bu tahminleme doğrusalmodele uygun değil
