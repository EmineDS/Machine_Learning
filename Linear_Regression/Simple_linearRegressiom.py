#basit doğrusal regresyon bir bağımlı bir bağımsız değerden oluşur bunu zaten biliyorduk
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data=pd.read_csv("Salary_Data.csv")
veri=data.copy()
print(veri)
X=veri["YearsExperience"]
Y=veri["Salary"]
#değişkenler doğrusal mı diye kontrol edelim.
plt.scatter(X,Y)
plt.show()
#doğrusal olduğu için şimdi regresyon yapımısı kurabiliriz
constant=sm.add_constant(X)
model=sm.OLS(Y,constant).fit()
print(model.summary())
#****** Eğer özel bir parametreyi çağırmak istiyorsak bu şekilde de kullanım yapılabilir.
lr=LinearRegression()
lr.fit(X.values.reshape(-1,1),Y.values.reshape(-1,1))
#coef_ bize x değerinin katsayısını intercept_ ise bize sabit değeri verir
print(lr.coef_,lr.intercept_)
#tahmin y değerleri de bu şekilde döndürülür.
print(lr.predict((X.values.reshape(-1,1))))
