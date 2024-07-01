import pandas as pd
data=pd.read_csv("./Mall_Customers.csv")
veri=data.copy()
pd.set_option("display.max_columns",None)

veri.drop(columns="CustomerID",axis=1,inplace=True)

print(veri.isnull().sum())
print(veri.dtypes)
veri.Gender=[1 if kod=="Male" else 0 for kod in veri.Gender]
print(veri)