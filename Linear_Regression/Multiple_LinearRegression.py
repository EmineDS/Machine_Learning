import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
data=pd.read_csv(".\\advertising.csv")
veri=data.copy()
print(veri)
#eksik gözlem var mı kontrol ettik
print(veri.isnull().sum())
#veri türlerini kontrol ettik.
print(veri.dtypes)
#satış bağımlı değişkenin diğer bağımsız değişkenlerle korelasyonuna bakalım
print(veri.corr()["Sales"])
#bu korelasyonları görselleştirelim
sns.pairplot(veri,kind="reg")
plt.show()
#şimdi de aykırı gözlem değerlerine bakalım
sns.boxplot(veri["TV"])# TV DE AYKIRI GÖZLEM DEĞERİ YOK
plt.show()
sns.boxplot(veri["Radio"])# Radşoda da  AYKIRI GÖZLEM DEĞERİ YOK
plt.show()
sns.boxplot(veri["Newspaper"])# Gazete de aykırı gözlem değerleri var
plt.show()
#Gazetedeki aykırı gözlem değerleri için baskılama yöntemi kullanalım
Q1=veri["Newspaper"].quantile(0.25)
Q3=veri["Newspaper"].quantile(0.75)
IQR=Q3-Q1
ustsınır=Q3+1.5*IQR
altsınır=Q1-1.5*IQR
#yalnızca ust sınrıın üzerinde aykırı değerler olduğu için alt sınırı kullanmadık
aykırı=veri["Newspaper"]>ustsınır
veri.loc[aykırı,"Newspaper"]=ustsınır
#modelimizi kuralım
X=veri[["TV","Radio","Newspaper"]]
Y=veri["Sales"]
constant=sm.add_constant(X)
model=sm.OLS(Y,constant).fit()
print(model.summary())
#gazete p değeri anlamsız olduğu için çıkaralım
X=veri[["TV","Radio"]]
Y=veri["Sales"]
constant=sm.add_constant(X)
model=sm.OLS(Y,constant).fit()
print(model.summary())
#MAKİNE ÖĞRENMESİ MODELİNE GEÇELİM
#data setini eğitim ve test seti olarak ayıralım
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20,random_state=42)#%20 lik test datası istedik
print(X_train)#kontrol edelim
lr=LinearRegression()
lr.fit(X_train,Y_train)#öğrenme işlemini gerçekleştirelim
print(lr.coef_)#iki bağımsız değişkenin katsayısını döndürür.
tahmin=lr.predict(X_test) #öğrenilen yapıyı test edelim
print(tahmin)
#tahmin değerlerini kendi asıl değerlerimizi karşılaştıralım
Y_test=Y_test.sort_index()#y test değerlerini düzgün bir grafik görmek için sıraladık
df=pd.DataFrame({"Gerçek":Y_test,"Tahmin":tahmin})
df.plot(kind="line")
plt.show()
#tahmin modelimizi oluşturduk :)