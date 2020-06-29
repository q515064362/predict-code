def datahandle():
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

  import warnings
  warnings.filterwarnings('ignore')

  electricity = pd.read_excel('C:/Users/E/Desktop/hourlyElectricityWithFeatures.xlsx')
  electricity = electricity.drop('startTime', 1).drop('endTime', 1)
  electricity = electricity.dropna()

  electricity.rename(columns={'Unnamed: 0':'Datetime'}, inplace=True)
  nonBlankColumns = ['Unnamed' not in s for s in electricity.columns]
  columns = electricity.columns[nonBlankColumns]
  electricity = electricity[columns]
  electricity.index.name = None
  electricity.head()

  electricity.set_index(["Datetime"], inplace=True)
  electricity[["dehumidification"]] = electricity[["dehumidification"]].apply(zscore)
  electricity[["windSpeed-m/s"]] = electricity[["windSpeed-m/s"]].apply(zscore)
  electricity[["solarRadiation-W/m2"]] = electricity[["solarRadiation-W/m2"]].apply(zscore)
  electricity[["pressure-mbar"]] = electricity[["pressure-mbar"]].apply(zscore)

  def replace(group): 
      median, std = group.median(), group.std() #Get the median and the standard deviation of every group 
      outliers = (group - median).abs() > 3*std # Subtract median from every member of each group. Take absolute values > 2std 
      group[outliers] = group.median().axis=0
      return group

  electricity[["dehumidification"]] = replace(electricity[["dehumidification"]])
  electricity[["windSpeed-m/s"]] = replace(electricity[["windSpeed-m/s"]])
  electricity[["solarRadiation-W/m2"]] = replace(electricity[["solarRadiation-W/m2"]])
  electricity[["pressure-mbar"]] =  replace(electricity[["pressure-mbar"]])
  def addHourlyTimeFeatures(df):
      df['hour'] = df.index.hour
      df['weekday'] = df.index.weekday
      df['day'] = df.index.dayofyear
      df['week'] = df.index.weekofyear    
      return df

  electricity = addHourlyTimeFeatures(electricity)
  electricity['day_type'] = np.zeros(len(electricity))
  electricity['day_type'][(electricity.index.dayofweek==5)|(electricity.index.dayofweek==6)] = 1

  # Set holidays to 1
  holidays = ['2014-01-01','2014-01-20','2014-05-26','2014-07-04','2014-09-01','2014-11-11','2014-11-27','2014-12-25','2013-01-01',
              '2013-01-21','2013-05-27','2013-07-04','2013-09-02','2013-11-11','2013-11-27','2013-12-25','2012-01-01','2012-01-16',
              '2012-05-28','2012-07-04','2012-09-03','2012-11-12','2012-11-22','2012-12-25']
  for i in range(len(holidays)):
      electricity['day_type'][electricity.index.date==np.datetime64(holidays[i])] = 1
  return electricity
