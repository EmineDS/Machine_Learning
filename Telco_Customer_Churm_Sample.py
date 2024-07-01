import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from lazypredict.Supervised import LazyClassifier
import numpy as np
data=pd.read_csv("./Telco-Customer-Churn.csv")
veri=data.copy()

pd.set_option("display.max_columns",None)
veri.drop(columns="customerID",axis=1,inplace=True)

print(veri.isnull().sum())
veri.TotalCharges=pd.to_numeric(veri.TotalCharges,errors="coerce")
print(veri.info())
for x in veri.select_dtypes("object").columns :
    veri[x]=LabelEncoder().fit_transform(veri[x])
veri.dropna(axis=0,inplace=True)
y=veri['Churn'].astype(int)
X=veri.drop(columns="Churn",axis=1)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


print(veri.info())
# clf=LazyClassifier(ignore_warnings=False)
# model,tahmin=clf.fit(X_train,X_test,y_train,y_test)
# print(model.sort_values(by="ROC AUC",ascending=False))
#                                  Accuracy  Balanced Accuracy  ROC AUC  F1 Score  \
# Model
# QuadraticDiscriminantAnalysis      0.77               0.77     0.77      0.78
# NearestCentroid                    0.72               0.76     0.76      0.74
# GaussianNB                         0.76               0.76     0.76      0.77
# BernoulliNB                        0.76               0.76     0.76      0.77
# LinearSVC                          0.82               0.74     0.74      0.82
# LinearDiscriminantAnalysis         0.82               0.74     0.74      0.81
# CalibratedClassifierCV             0.82               0.74     0.74      0.82
# LogisticRegression                 0.82               0.74     0.74      0.81
# RidgeClassifier                    0.82               0.73     0.73      0.81
# RidgeClassifierCV                  0.82               0.73     0.73      0.81
# SGDClassifier                      0.80               0.73     0.73      0.80
# LGBMClassifier                     0.80               0.72     0.72      0.80
# AdaBoostClassifier                 0.80               0.72     0.72      0.80
# SVC                                0.81               0.71     0.71      0.80
# from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
# Quad=QuadraticDiscriminantAnalysis()
# parametreler={"priors": [None],"reg_param": np.linspace(0, 5, num=10),"store_covariance":[True,False]}
# grid=GridSearchCV(Quad,param_grid=parametreler,cv=10,n_jobs=-1)
# grid.fit(X_train,y_train)
# print(grid.best_params_)
# # {'priors': None, 'reg_param': 0.6666666666666666, 'store_covariance': True}
# Quad=QuadraticDiscriminantAnalysis(priors=None,reg_param=0.5555555555555556,store_covariance=True)
# Quad.fit(X_train,y_train)
# tahmin=Quad.predict(X_test)
# acs=accuracy_score(y_test,tahmin)
# print(acs*100)
# 75.12437810945273

# from sklearn.neighbors import NearestCentroid
# parametreler={"metric":["euclidean", "manhattan"],"shrink_threshold":range(-100,100)}
# nc=NearestCentroid(metric="euclidean",shrink_threshold=-19)
# grid=GridSearchCV(nc,param_grid=parametreler,cv=10,n_jobs=-1)
# grid.fit(X_train,y_train)
# print(grid.best_params_)
# {'metric': 'euclidean', 'shrink_threshold': -19}
# nc.fit(X_train,y_train)
# tahmin=nc.predict(X_test)
# acs=accuracy_score(y_test,tahmin)
# print(acs*100)
# 76.83013503909027

# from sklearn.naive_bayes import GaussianNB
# gb=GaussianNB(priors=None,var_smoothing=0.9831983198319832)
# gb.fit(X_train,y_train)
# parametreler={"var_smoothing":np.linspace(0.0000000000000001, 1, num=10000)}
# grid=GridSearchCV(gb,param_grid=parametreler,cv=10,n_jobs=-1)
# grid.fit(X_train,y_train)
# print(grid.best_params_)
# {'var_smoothing': 0.9831983198319832}
# tahmin=gb.predict(X_test)
# acs=accuracy_score(y_test,tahmin)
# print(acs*100)
# 75.1954513148543

# from sklearn.naive_bayes import BernoulliNB
# bn=BernoulliNB(alpha=1457, binarize= 0, fit_prior= True)
# bn.fit(X_train,y_train)
# paramereler={"alpha":np.arange(0,10000),"binarize":np.arange(0,1000,999),"fit_prior":[True,False]}
# grid=GridSearchCV(bn,param_grid=paramereler,cv=10,n_jobs=-1)
# grid.fit(X_train,y_train)
# print(grid.best_params_)
# tahmin=bn.predict(X_test)
# acs=accuracy_score(y_test,tahmin)
# print(acs*100)
# 75.97725657427151
from sklearn.svm import LinearSVC
lsvc=LinearSVC()
lsvc.fit(X_train,y_train)
parametreler={"penalty":["l1", "l2"],"dual":["auto","bool"],"tol":np.arange(0.000001,1),"C":np.arange(0,10),"multi_class":["ovr","crammer_singer"],"fit_intercept":[True,False],"intercept_scaling":np.arange(0,50)}
              # "class_weight":[dict,"balanced",None],"verbose":range(-100,100),"random_state":range(0,100),"max_iter":range(10,1000)}
grid=GridSearchCV(lsvc,param_grid=parametreler,n_jobs=-1,cv=10)
grid.fit(X_train,y_train)
print(grid.best_params_)