# Load libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot
from pandas import read_csv
import seaborn as sns
from numpy import arange
from pandas import  set_option
from pandas.plotting import scatter_matrix

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from scipy.stats import zscore

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor

from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error


%matplotlib inline

import handle_data
electricity=handle_data.datahandle()

elect_train = pd.DataFrame(data=electricity1, index=np.arange('2012-01-01 00:00', '2014-03-30 00:00', dtype='datetime64[h]')).dropna()
elect_test = pd.DataFrame(data=electricity1, index=np.arange('2014-04-01 00:00', '2014-05-01 00:00', dtype='datetime64[h]')).dropna()

XX_elect_train = elect_train.drop('electricity-kWh', axis = 1)
XX_elect_test = elect_test.drop('electricity-kWh', axis = 1)

YY_elect_train = elect_train['electricity-kWh']
YY_elect_test = elect_test['electricity-kWh']

import train_data
gbr,ETR,GBR2,ETR2,predictions_GBR,predictions_ETR,predictions_ETR2,predictions_GBR2,rescaledX_validation,rescaledX_validation2=train_data.datatrain((XX_elect_train,XX_elect_test,YY_elect_train,YY_elect_test))

import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']  

from matplotlib import style
print(plt.style.available)

fig,ax = plt.subplots(1, 1,figsize=(40,20))
ax.xaxis.set_major_formatter(mdate.DateFormatter('%m-%d %H'))#设置时标签显示格式
line1, =plt.plot(XX_elect_test.index[310:390], YY_elect_test[310:390], label='Actual consumption', color='k',linewidth='3')
line2, =plt.plot(XX_elect_test.index[310:390], predictions_GBR[310:390], label='GBR with mixed feature' , color='r',linewidth='3')
line3, =plt.plot(XX_elect_test.index[310:390], predictions_ETR[310:390], label='ETR with mixed feature', color='b',linewidth='3')
#line4, =plt.plot(XX_elect_test.index[310:390], predictions_RF[310:390], label='RF')

#line5, =plt.plot(XX_elect_test.index[310:390], predictions_GBR2[310:390], label='GBR2')
line6, =plt.plot(XX_elect_test.index[310:390], predictions_ETR2[310:390], label='ETR with single feature', color='y',linewidth='3')
line7, =plt.plot(XX_elect_test.index[310:390], predictions_GBR2[310:390], label='GBR with single feature', color='g',linewidth='3')

plt.xticks(size=30)
plt.yticks(size=30)
plt.xlabel('Time',fontsize=33)
plt.ylabel('Electricity usage (kWh)',fontsize=30)
plt.title('Electricity usage by different models and feature selection algorithms',fontsize=35)
plt.legend([line1, line2,line3,line6,line7], ['Actual consumption', 'GBR with mixed features','ETR with mixed features','ETR with single features','GBR with single features'],fontsize=25)
plt.style.use('ggplot')
plt.show()

from sklearn import metrics

print ("GBR2 The test score R2(GBR_S): ",GBR2.score(rescaledX_validation2[310:390],YY_elect_test2[310:390]))
GBR_S_mse_test=np.sum((predictions_GBR2[310:390]-YY_elect_test2[310:390])**2/len(YY_elect_test2[310:390]))
GBR_S_MAPE_test=np.sum((abs(predictions_GBR2[310:390]-YY_elect_test2[310:390])/YY_elect_test2[310:390]))*100/len(YY_elect_test2[310:390])
print ("RF2 The test score MAPE: ",GBR_S_MAPE_test)
print ("RF2 The test score MSE: ",GBR_S_mse_test)
GBR_S_rmse_test=GBR_S_mse_test**0.5
print ("RF2 The test score RMSE: ",GBR_S_rmse_test)
print('\n')

print ("ETR The test score R2(etr_m): ",ETR.score(rescaledX_validation[310:390],YY_elect_test[310:390]))

etr_m_mse_test=np.sum((predictions_ETR[310:390]-YY_elect_test[310:390])**2/len(YY_elect_test[310:390]))
etr_m_MAPE_test=np.sum((abs(predictions_ETR[310:390]-YY_elect_test[310:390])/YY_elect_test[310:390]))*100/len(YY_elect_test[310:390])
print ("ETR The test score MAPE: ",etr_m_MAPE_test)
print ("ETR The test score MSE: ",etr_m_mse_test)
etr_m_rmse_test=etr_m_mse_test**0.5
print ("ETR The test score RMSE: ",etr_m_rmse_test)
print('\n')

print ("GBR The test score R2(gbr_M): ",gbr.score(rescaledX_validation[310:390],YY_elect_test[310:390]))
#gbr_M_mse_test=np.sum((predictions_GBR[310:390]-YY_elect_test[310:390])**2/len(YY_elect_test[310:390]))
gbr_M_mse_test=metrics.mean_squared_error(YY_elect_test[310:390], predictions_GBR[310:390])
gbr_M_MAPE_test=np.sum((abs(predictions_GBR[310:390]-YY_elect_test[310:390])/YY_elect_test[310:390]))*100/len(YY_elect_test[310:390])
print ("GBR The test score MAPE: ",gbr_M_MAPE_test)
print ("GBR The test score MSE: ",gbr_M_mse_test)
gbr_M_rmse_test=gbr_M_mse_test**0.5
print ("GBR The test score RMSE: ",gbr_M_rmse_test)

print('\n')

print ("ETR2 The test score R2(ETR_s): ",ETR2.score(rescaledX_validation2[310:390],YY_elect_test2[310:390]))

ETR_s_mse_test=np.sum((predictions_ETR2[310:390]-YY_elect_test2[310:390])**2/len(YY_elect_test2[310:390]))
ETR_s_MAPE_test=np.sum((abs(predictions_ETR2[310:390]-YY_elect_test2[310:390])/YY_elect_test2[310:390]))*100/len(YY_elect_test2[310:390])
print ("ETR2 The test score MAPE: ",ETR_s_MAPE_test)
print ("ETR2 The test score MSE: ",ETR_s_mse_test)
ETR_s_rmse_test=ETR_s_mse_test**0.5
print ("ETR2 The test score RMSE: ",ETR_s_rmse_test)

import numpy as np

#import matplotlib.pyplot as plt
ticks = [i for i in range(3)]
names=['MAPE','MES','RMSE']



GBR_S=[GBR_S_mse_test,GBR_S_MAPE_test,GBR_S_rmse_test]
etr_m=[etr_m_mse_test,etr_m_MAPE_test,etr_m_rmse_test]
ETR_s=[ETR_s_mse_test,ETR_s_MAPE_test,ETR_s_rmse_test]
gbr_M=[gbr_M_mse_test,gbr_M_MAPE_test,gbr_M_rmse_test]

pyplot.figure(figsize=(15, 12), dpi=100) 
x=ticks
y1=GBR_S
y2=etr_m
y3=ETR_s
y4=gbr_M

ind = np.arange(len(y1))  # the x locations for the groups
width = 0.2  # the width of the bars

print(ind)
 
fig, ax = pyplot.subplots(figsize=(15, 12), dpi=60)
rects1 = ax.bar(ind - width, y1, width, label='GBR with single feature')
rects2 = ax.bar(ind , y2, width, label='ETR with mixed feature')
rects3 = ax.bar(ind +width, y3, width, label='ETR with single feature')
rects4 = ax.bar(ind +width*2, y4, width, label='GBR with mixed feature')
plt.legend()  # 显示图例

pyplot.xlabel('Feature',size=15)
pyplot.ylabel('Importance value',size=15) 
# 添加标题 
pyplot.title('Feature importance analysis',size=20)    
    
def autolabel(p):
    for i in p:
        height = i.get_height()
        pyplot.text(i.get_x()-0.06+i.get_width()/2., height+10, '%.1f' % height)
autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)
pyplot.xticks(ticks, names)
pyplot.show()


GBR_M_err=abs(YY_elect_test-predictions_GBR)
ETR_M_err=abs(YY_elect_test-predictions_ETR)
GBR_S_err=abs(YY_elect_test-predictions_GBR2)
ETR_S_err=abs(YY_elect_test-predictions_ETR2)

plt.style.use('seaborn-white')
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False
plt.grid(False)

fig,ax = plt.subplots(1, 1,figsize=(40,20))

#line1, =plt.plot(XX_elect_test.index[310:390], GBR_M_err[310:390], label='Actual consumption', color='k',linewidth='3')
line2, =plt.plot(XX_elect_test.index[310:390], GBR_M_err[310:390], label='GBR with mixed feature' , color='r',linewidth='3')
line3, =plt.plot(XX_elect_test.index[310:390], ETR_M_err[310:390], label='ETR with mixed feature', color='b',linewidth='3')
#line4, =plt.plot(XX_elect_test.index[310:390], predictions_RF[310:390], label='RF')

#line5, =plt.plot(XX_elect_test.index[310:390], predictions_GBR2[310:390], label='GBR2')
line6, =plt.plot(XX_elect_test.index[310:390], ETR_S_err[310:390], label='ETR with single feature', color='y',linewidth='3')
line7, =plt.plot(XX_elect_test.index[310:390], GBR_S_err[310:390], label='GBR with single feature', color='g',linewidth='3')

plt.xticks(size=30)
plt.yticks(size=30)
plt.xlabel('Time',fontsize=33)
plt.ylabel('Absolute error',fontsize=30)
plt.title('Absolute error of different models and feature selection algorithms (kWh)',fontsize=35)
plt.legend([ line2,line3,line6,line7], [ 'GBR with mixed features','ETR with mixed features','ETR with single features','GBR with single features'],fontsize=25)
plt.style.use('ggplot')
plt.show()

#Plot Observed vs. Linear Regression predicted usage.

fig= plt.figure(figsize=(6,6))
plt.style.use('seaborn-white')
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False
plt.grid(None)
plt.plot(YY_elect_test, YY_elect_test, c='k')
plt.scatter(YY_elect_test, predictions_GBR, c='g')
plt.grid(None)
#fig.set_facecolor('blueviolet')
#ax.grid(True, linestyle='-.')
plt.xlabel(u'实际的用电负荷量 (kWh)')
plt.ylabel(u"预测的用电负荷量 (kWh)")
#plt.ylabel("Predicted Elec. Usage : $\hat{Y}_i$(kWh)")
#plt.title("Energy vs Predicted Elec.: $Y_i$ vs $\hat{Y}_i$")

#Plot Observed vs. Linear Regression predicted usage.
fig = plt.figure(figsize=(6,6))  #GBR_M
plt.grid(None)
plt.plot(YY_elect_test, YY_elect_test, c='k')
plt.scatter(YY_elect_test, predictions_GBR, c='g')
plt.grid(None)
plt.xlabel(u'实际的用电负荷量 (kWh)')
plt.ylabel(u"预测的用电负荷量 (kWh)")
#plt.ylabel("Predicted Elec. Usage : $\hat{Y}_i$(kWh)")
#plt.title("Energy vs Predicted Elec.: $Y_i$ vs $\hat{Y}_i$")

#Plot Observed vs. Linear Regression predicted usage.
fig = plt.figure(figsize=(6,6)) # GBR_S
plt.grid(None)
plt.plot(YY_elect_test, YY_elect_test, c='k')
plt.scatter(YY_elect_test, predictions_RF2, c='g')
plt.grid(None)
plt.xlabel(u'实际的用电负荷量 (kWh)')
plt.ylabel(u"预测的用电负荷量 (kWh)")
#plt.ylabel("Predicted Elec. Usage : $\hat{Y}_i$(kWh)")
#plt.title("Energy vs Predicted Elec.: $Y_i$ vs $\hat{Y}_i$")

#Plot Observed vs. Linear Regression predicted usage.
fig = plt.figure(figsize=(6,6)) # etr_M
plt.grid(None)
plt.plot(YY_elect_test, YY_elect_test, c='k')
plt.scatter(YY_elect_test, predictions_ETR, c='g')
plt.grid(None)
plt.xlabel(u'实际的用电负荷量 (kWh)')
plt.ylabel(u"预测的用电负荷量 (kWh)")
#plt.ylabel("Predicted Elec. Usage : $\hat{Y}_i$(kWh)")
#plt.title("Energy vs Predicted Elec.: $Y_i$ vs $\hat{Y}_i$")

#Plot Observed vs. Linear Regression predicted usage.
fig = plt.figure(figsize=(6,6)) # etr_S
plt.grid(None)
plt.plot(YY_elect_test, YY_elect_test, c='k')
plt.scatter(YY_elect_test, predictions_ETR2, c='g')
plt.grid(None)
plt.xlabel(u'实际的用电负荷量 (kWh)')
plt.ylabel(u"预测的用电负荷量 (kWh)")
#plt.ylabel("Predicted Elec. Usage : $\hat{Y}_i$(kWh)")
#plt.title("Energy vs Predicted Elec.: $Y_i$ vs $\hat{Y}_i$")

#gbr_M
a=0.0
b=0.0
c=0.0
d=0.0
e=0.0

for i in range(len(YY_elect_test)):
    num=abs(YY_elect_test[i]-predictions_GBR[i])
    p=num/YY_elect_test[i]
    if p<=0.01:
        a+=1
    if p<=0.025:
        b+=1
    if p<=0.05:
        c+=1
    if p<=0.1:
        d+=1
    if p<=0.25:
        e+=1

print('gbr_M')
print('%1的  {:.2%}'.format(a/80))
print('%2.5的  {:.2%}'.format(b/80))
print('%5的  {:.2%}'.format(c/80))
print('%10的  {:.2%}'.format(d/80))
print('%25的  {:.2%}'.format(e/80))

#GBR_S
a=0.0
b=0.0
c=0.0
d=0.0
e=0.0

for i in range(len(YY_elect_test)):
    num=abs(YY_elect_test[i]-predictions_RF2[i])
    p=num/YY_elect_test[i]
    if p<=0.01:
        a+=1
    if p<=0.025:
        b+=1
    if p<=0.05:
        c+=1
    if p<=0.1:
        d+=1
    if p<=0.25:
        e+=1

print('GBR_S')
print('%1的  {:.2%}'.format(a/80))
print('%2.5的  {:.2%}'.format(b/80))
print('%5的  {:.2%}'.format(c/80))
print('%10的  {:.2%}'.format(d/80))
print('%25的  {:.2%}'.format(e/80))

#etr_m
a=0.0
b=0.0
c=0.0
d=0.0
e=0.0

for i in range(len(YY_elect_test)):
    num=abs(YY_elect_test[i]-predictions_ETR[i])
    p=num/YY_elect_test[i]
    if p<=0.01:
        a+=1
    if p<=0.025:
        b+=1
    if p<=0.05:
        c+=1
    if p<=0.1:
        d+=1
    if p<=0.25:
        e+=1

print('etr_m')
print('%1的  {:.2%}'.format(a/80))
print('%2.5的  {:.2%}'.format(b/80))
print('%5的  {:.2%}'.format(c/80))
print('%10的  {:.2%}'.format(d/80))
print('%25的  {:.2%}'.format(e/80))

#ETR_S
a=0.0
b=0.0
c=0.0
d=0.0
e=0.0

for i in range(len(YY_elect_test)):
    num=abs(YY_elect_test[i]-predictions_ETR2[i])
    p=num/YY_elect_test[i]
    if p<=0.01:
        a+=1
    if p<=0.025:
        b+=1
    if p<=0.05:
        c+=1
    if p<=0.1:
        d+=1
    if p<=0.25:
        e+=1

print('ETR_S')
print('%1的  {:.2%}'.format(a/80))
print('%2.5的  {:.2%}'.format(b/80))
print('%5的  {:.2%}'.format(c/80))
print('%10的  {:.2%}'.format(d/80))
print('%25的  {:.2%}'.format(e/80))

# -*- coding: UTF-8 -*-
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

#这里导入你自己的数据
#......
#......
#x_axix，train_pn_dis这些都是长度相同的list()

fig = plt.figure(figsize=(8,6))

x_axix=['<1%','<2.5%','<5%','<10%','<25%']
GBR_M=[13.6,30.0,56.2,85.0,100]
ETR_M=[8.7,31.25,50.0,73.7,100]
GBR_S=[6.2,16.25,23.7,41.2,85]
ETR_S=[5,16.2,27.5,46.2,86.2]


#开始画图
#sub_axix = filter(lambda x:x%200 == 0, x_axix)
#plt.title('Percentage of cases when error falls into range')
plt.plot(x_axix, GBR_M, color='red', label=u'复合特征选择和GBDT混合模型',marker='*')
plt.plot(x_axix, ETR_M, color='blue', label=u'复合特征选择和ETR混合模型',marker='*')
plt.plot(x_axix, ETR_S,  color='y', label=u'单一特征选择和GBDT混合模型',marker='*')
plt.plot(x_axix, GBR_S, color='green', label=u'单一特征选择和ETR混合模型',marker='*')
plt.legend(fontsize=16,loc='upper left') # 显示图例
plt.grid(axis="y",linestyle='-.')
plt.xlabel(u'百分比区间',size=22)
plt.ylabel(u'百分比值 (%)',size=22)
plt.tick_params(labelsize=20)
plt.show()
#python 一个折线图绘制多个曲线

