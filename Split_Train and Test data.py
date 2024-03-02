import pandas as pd
from sklearn.model_selection import train_test_split
data=pd.read_excel("./Training_Test.xlsx")
#pd.set_option('display.max_rows', None)
print(data)
#parçalama işlmeini yapmadan önce değişkenleri birbirinden ayırlaım
y=data["Y"]
X=data[["X1","X2"]]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)#%20 sinin test verisi olacağını belirttik
print(X_train)
print(X_test)
#VERİ SETİMİZİ EĞİTİM VE TEST VERİ SETİ OLARAK PARÇALADIK.
#EĞİTİM VERİLERİ MODELİ EĞİTİRKEN TEST VERİLEİR MODELİN BAŞARISINI TEST EDECEK.

#ANCAK BİR SORUN VAR Kİ;
#Her çalıştırma işlminde alınan veriler değişecek.
#her çalışmada yeniden cx verilerinin %80 nini rassal olarak alacak
#bunu önlemek iin bir parametre daha ekliyoruz.
# Random_state parametresi isteiğiniz sayıyı girebilirsiniz ancak 42 geleneksel olarak tercih edilir.
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
print(X_train)
print(X_test)