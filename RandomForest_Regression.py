import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
data=pd.read_csv("./Position_Salaries.csv")
veri=data.copy()
print(veri)
y=veri["Salary"]
X=veri["Level"]

X=np.array(X).reshape(-1,1)
y=np.array(y).reshape(-1,1)
#karar ağacı regresyonu
dtmodel=DecisionTreeRegressor(random_state=0)
dtmodel.fit(X,y)
dttahmin=dtmodel.predict(X)

#random forest regresyon
rfmodel=RandomForestRegressor(random_state=0)
rfmodel.fit(X,y)
rftahmin=rfmodel.predict(X)

plt.scatter(X,y,color="red")
plt.plot(X,dttahmin,color="blue",label="karar ağacı")
plt.plot(X,rftahmin,color="green",label="Random Forest")
plt.title("overfitting düzeltildi")
plt.legend()
plt.show()
