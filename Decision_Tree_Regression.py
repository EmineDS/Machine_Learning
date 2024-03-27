import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor,plot_tree
data=pd.read_csv(R"./Position_Salaries.csv")
veri=data.copy()
print(veri)

y=veri["Salary"]
y=np.array(y).reshape(-1,1)
X=veri["Level"]
X=np.array(X).reshape(-1,1)
print(X,y)

dtr=DecisionTreeRegressor(random_state=0,max_leaf_nodes=3)
#max_leaf_nodes elimizde kaç tane sonuç seçeneği olduğunu belirlememizi sağlar
dtr.fit(X,y) #train data setleriyle eğiticez
tahmin=dtr.predict(X)

plt.scatter(X,y,color="red")
plt.plot(X,tahmin)
#tahmin değerleirmiz gerçeğe ne adar uygun ona bakıyoruz
plt.show()
plt.figure(figsize=(20,10),dpi=100)
plot_tree(dtr,feature_names=X,class_names=y,rounded=True,filled=True)
plt.show()