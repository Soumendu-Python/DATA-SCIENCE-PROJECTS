# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 20:05:04 2021

@author: soumendu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
os.chdir(r'C:\Post Graduate Course in Data Analytics\KAGGLE DATASETS & SCRIPTS\OCEAN TEMPERATURE PREDICTION')

df=pd.read_csv('bottle.csv')

df=df[['Salnty','T_degC']]

df.info()
df.isnull().sum()
df.shape

df1=df[df.T_degC.isnull()]

df1.isnull().sum()

df1=df1[df1.Salnty.notnull()]


df=df[df.T_degC.notnull()]

df=df[df.Salnty.notnull()]

df.isnull().sum()

df2=df[:][:1000]

plt.scatter(x=df2['T_degC'],y=df2['Salnty'])
plt.title('Scatter PLot showing relation between Salinity and Temperature of Ocean Water')
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#Linear Regression
x=df2['Salnty'].values.reshape(-1,1)
y=df2['T_degC'].values.reshape(-1,1)

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25,random_state=47)

lr=LinearRegression()
lrmodel=lr.fit(xtrain,ytrain)
ypred_lr=lrmodel.predict(xtest)

from sklearn.metrics import mean_squared_error,r2_score

mse_lr=mean_squared_error(ytest,ypred_lr)
rmse_lr=np.sqrt(mse_lr)
print(rmse_lr)

r2_lr=r2_score(ytest,ypred_lr)
print(r2_lr)

from yellowbrick.regressor import prediction_error,residuals_plot

x=df2['Salnty'].values.reshape(-1,1)
y=df2['T_degC']
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25,random_state=47)

visualizer=prediction_error(LinearRegression(),xtrain,ytrain,xtest,ytest)
viz=residuals_plot(LinearRegression(),xtrain,ytrain,xtest,ytest)

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso,Ridge

#Lasso Regression
x=df2['Salnty'].values.reshape(-1,1)
y=df2['T_degC'].values.reshape(-1,1)

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25,random_state=47)

ls=Lasso(alpha=0.1)
lsmodel=ls.fit(xtrain,ytrain)
ypred_ls=lsmodel.predict(xtest)

mse_ls=mean_squared_error(ytest,ypred_ls)
rmse_ls=np.sqrt(mse_ls)
print(rmse_ls)

r2_ls=r2_score(ytest,ypred_ls)
print(r2_ls)

x=df2['Salnty'].values.reshape(-1,1)
y=df2['T_degC']
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25,random_state=47)

visualizer=prediction_error(Lasso(alpha=0.1),xtrain,ytrain,xtest,ytest)
viz=residuals_plot(Lasso(alpha=0.1),xtrain,ytrain,xtest,ytest)

#Ridge Regression
x=df2['Salnty'].values.reshape(-1,1)
y=df2['T_degC'].values.reshape(-1,1)

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25,random_state=47)

rd=Ridge()
rdmodel=rd.fit(xtrain,ytrain)
ypred_rd=rdmodel.predict(xtest)

mse_rd=mean_squared_error(ytest,ypred_rd)
rmse_rd=np.sqrt(mse_rd)
print(rmse_rd)

r2_rd=r2_score(ytest,ypred_rd)
print(r2_rd)

x=df2['Salnty'].values.reshape(-1,1)
y=df2['T_degC']
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25,random_state=47)

visualizer=prediction_error(Ridge(),xtrain,ytrain,xtest,ytest)
viz=residuals_plot(Ridge(),xtrain,ytrain,xtest,ytest)

#Decision Tree Regression
x=df2['Salnty'].values.reshape(-1,1)
y=df2['T_degC'].values.reshape(-1,1)

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25,random_state=47)

dtr=DecisionTreeRegressor()
dtrmodel=dtr.fit(xtrain,ytrain)
ypred_dtr=dtrmodel.predict(xtest)

mse_dtr=mean_squared_error(ytest,ypred_dtr)
rmse_dtr=np.sqrt(mse_dtr)
print(rmse_dtr)

r2_dtr=r2_score(ytest,ypred_dtr)
print(r2_dtr)

x=df2['Salnty'].values.reshape(-1,1)
y=df2['T_degC']
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25,random_state=47)

visualizer=prediction_error(DecisionTreeRegressor(),xtrain,ytrain,xtest,ytest)
viz=residuals_plot(DecisionTreeRegressor(),xtrain,ytrain,xtest,ytest)

#Random Forest Regression
x=df2['Salnty'].values.reshape(-1,1)
y=df2['T_degC'].values.reshape(-1,1)

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25,random_state=47)

rfr=RandomForestRegressor(n_estimators=100)
rfrmodel=rfr.fit(xtrain,ytrain)
ypred_rfr=rfrmodel.predict(xtest)

mse_rfr=mean_squared_error(ytest,ypred_rfr)
rmse_rfr=np.sqrt(mse_rfr)
print(rmse_rfr)

r2_rfr=r2_score(ytest,ypred_rfr)
print(r2_rfr)

x=df2['Salnty'].values.reshape(-1,1)
y=df2['T_degC']
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25,random_state=47)

visualizer=prediction_error(RandomForestRegressor(n_estimators=100),xtrain,ytrain,xtest,ytest)
viz=residuals_plot(RandomForestRegressor(n_estimators=100),xtrain,ytrain,xtest,ytest)

#XGB Regression
from xgboost import XGBRegressor
x=df2['Salnty'].values.reshape(-1,1)
y=df2['T_degC'].values.reshape(-1,1)

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25,random_state=47)

xgb=XGBRegressor(n_estimators=100,learning_rate=0.1)
xgbmodel=xgb.fit(xtrain,ytrain)
ypred_xgb=xgbmodel.predict(xtest)

mse_xgb=mean_squared_error(ytest,ypred_xgb)
rmse_xgb=np.sqrt(mse_xgb)
print(rmse_xgb)

r2_xgb=r2_score(ytest,ypred_xgb)
print(r2_xgb)

x=df2['Salnty'].values.reshape(-1,1)
y=df2['T_degC']
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25,random_state=47)

visualizer=prediction_error(XGBRegressor(n_estimators=100,learning_rate=0.1),xtrain,ytrain,xtest,ytest)
viz=residuals_plot(XGBRegressor(n_estimators=100,learning_rate=0.1),xtrain,ytrain,xtest,ytest)

comparison_dict={'Regression Models':['Linear Regression Model','Lasso Model',
                           'Ridge Model','Decision Tree Regression Model',
                           'Random Forest Regression Model','XGBoost Regression Model'],
                 'RMSE Value':[rmse_lr,rmse_ls,rmse_rd,rmse_dtr,
                               rmse_rfr,rmse_xgb],
                 'R2 Score':[r2_lr,r2_ls,r2_rd,r2_dtr,r2_rfr,r2_xgb]}

comparison_df=pd.DataFrame(comparison_dict)

unseen_value=df1['Salnty'].values.reshape(-1,1)
prediction=xgbmodel.predict(unseen_value)
df1['T_degC']=prediction

plt.scatter(x=df1['T_degC'],y=df1['Salnty'])
plt.title('Scatter PLot showing relation between Salinity and Temperature of Ocean Water in Unseen Data')
plt.show()
