#!/usr/bin/env python
# coding: utf-8

# In[ ]:


--> Data description <-- 


# In[2]:


import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[18]:


#load dataset   https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho
data = pd.read_csv('car data.csv')


# In[19]:


data.dtypes.value_counts()


# In[20]:


data.head(10)


# In[21]:


data.shape


# In[22]:


data.describe()


# In[23]:


#There is no missing values
data.isnull().sum()


# In[ ]:


--> Changing all categorical values into numerical <--


# In[24]:


mask = data.dtypes == np.object
cat_col = data.columns[mask]
#All categorical values 
cat_col


# In[25]:


from sklearn.preprocessing import LabelEncoder

label_en = LabelEncoder()
for cat in cat_col:
    label_en.fit(data[cat].drop_duplicates())
    data[cat] = label_en.transform(data[cat])


# In[26]:


data.head(10)


# In[ ]:


--> Correlations <-- 


# In[30]:


#Heat Map
plt.figure(dpi=200)
sns.heatmap(np.round(data.corr(),1), annot=True, cmap="Blues")
plt.show()


# In[31]:


#sorted features according to the correlation with Selling price feature 

data.corr()['Selling_Price'].sort_values(ascending = False)


# In[32]:


The strongest correlation was between Selling price and Present price 


# In[33]:


--> Normality Testing <--


# In[35]:


data.Selling_Price.hist()


# In[36]:


# D'Agostino K^2 test for normality 
from scipy.stats.mstats import normaltest
normaltest(data.Selling_Price.values)


# In[ ]:


#Our data is not normally distrtibuted as p-value is far away from 0.05
#Therefore we have to apply transformation to check which technique is closest to noramlity 


# In[40]:


#Square root transformation 
sqrt_SP = np.sqrt(data.Selling_Price)
plt.hist(sqrt_SP)

#Normality test
sqrt_test_SP = normaltest(sqrt_SP.values)
sqrt_test_SP


# In[41]:


#Log transformation 
log_SP = np.log(data.Selling_Price)
plt.hist(log_SP)

#Normality test
log_test_SP = normaltest(log_SP.values)
log_test_SP


# In[42]:


#boxcox transformation
from scipy.stats import boxcox


# In[44]:


bc = boxcox(data.Selling_Price)
boxcox_medv = bc[0]
lam = bc[1]


# In[45]:


plt.hist(boxcox_medv)
bc_test = normaltest(boxcox_medv)
bc_test
#as we can see that this technique brought us a closer result to the normal distribution 


# In[47]:


d_transf = {'Transformation' : ['SQRT', "LOG", "BOXCOX" ], 'p-value': [sqrt_test_SP[1], log_test_SP[1], bc_test[1]]}
df_ = pd.DataFrame(data=d_transf)
df_


# In[ ]:


#We can state that BOXCOX transformation is the closest to the normal distribution 


# In[48]:


# --> Regression Models <--


# In[49]:


# Importing Libraries
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


# In[52]:


car_data = data

mask = car_data.dtypes == np.object
cat_col = car_data.columns[mask]

#categorical data into numerical 
lab_e = LabelEncoder()
for cat in cat_col:
    lab_e.fit(car_data[cat].drop_duplicates())
    car_data[cat] = lab_e.transform(car_data[cat])

#Drop output variable and lowest cooraleted
X = car_data.drop(['Selling_Price', 'Seller_Type'], axis=1)
y = car_data.Selling_Price

# Polymonial Features
pf = PolynomialFeatures(degree=2, include_bias=False)
X_pf = pf.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_pf, y,
                                                   test_size=0.3, random_state=42)


# In[ ]:


# --> Application of regression models <--


# In[53]:


# Importing Libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import KFold, cross_val_predict, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline


# In[54]:


#load data
car_data = pd.read_csv('car data.csv')
car_data.head()


# In[56]:


#select categorical data
mask = car_data.dtypes == np.object
cat_col = car_data.columns[mask]

lab_e = LabelEncoder()
for cat in cat_col:
    lab_e.fit(car_data[cat].drop_duplicates())
    car_data[cat] = lab_e.transform(car_data[cat])
    
#Drop output variable and lowest cooraleted
X = car_data.drop(['Selling_Price', 'Seller_Type'], axis=1)
y = car_data.Selling_Price

#create folds 

kfolds = KFold(shuffle=True, random_state=42, n_splits=3) 


# In[60]:


# Vanilla Linear Regression

estimator = Pipeline([("scaler", StandardScaler()),
        ("polynomial_features", PolynomialFeatures()),
        ("linear_regression", LinearRegression())])

params = {
    "polynomial_features__degree": range(3),
}

grid = GridSearchCV(estimator, params, cv=kfolds)
grid.fit(X, y)
grid.best_score_, grid.best_params_


# In[61]:


best_estimator = Pipeline([
                    ("Scaler", StandardScaler()),
                    ("make_higher_degree", PolynomialFeatures(degree=1)),
                    ("vanilla_regression", LinearRegression())])

best_estimator.fit(X, y)
best_estimator.score(X, y)


# In[64]:


estimator.get_params().keys()


# In[66]:


#Lasso Regrassion 

estimator = Pipeline([("scaler", StandardScaler()),
        ("polynomial_features", PolynomialFeatures()),
        ("lasso_regression", Lasso())])

params = {
    "polynomial_features__degree": [1, 2, 3],
    "lasso_regression__alpha": np.geomspace(0.01, 30, 50)
}


grid = GridSearchCV(estimator, params, cv=kfolds)
grid.fit(X, y)
grid.best_score_, grid.best_params_


# In[74]:


best_lasso_estimator = Pipeline([
                    ("scaler", StandardScaler()),
                    ("make_higher_degree", PolynomialFeatures(degree=2)),
                    ("lasso_regression", Lasso(alpha=0.083))])

best_lasso_estimator.fit(X, y)
best_lasso_estimator.score(X, y)


# In[75]:


# Ridge regression
estimator = Pipeline([("scaler", StandardScaler()),
        ("polynomial_features", PolynomialFeatures()),
        ("ridge_regression", Ridge())])

params = {
    "polynomial_features__degree": [1, 2, 3],
    "ridge_regression__alpha": np.geomspace(2, 30, 20)
}

grid = GridSearchCV(estimator, params, cv=kfolds)
grid.fit(X, y)
grid.best_score_, grid.best_params_


# In[76]:


best_ridge_estimator = Pipeline([
                    ("scaler", StandardScaler()),
                    ("make_higher_degree", PolynomialFeatures(degree=1)),
                    ("ridge_regression", Ridge(alpha=16.96))])

best_ridge_estimator.fit(X, y)
best_ridge_estimator.score(X, y)


# In[77]:


from sklearn.metrics import mean_squared_error

def rmse(ytrue, ypredicted):
    return np.sqrt(mean_squared_error(ytrue, ypredicted))


# In[79]:


pf = PolynomialFeatures(degree=2)
s = StandardScaler()

X_pf = pf.fit_transform(X)
X_s = s.fit_transform(X_pf)
X_train, X_test, y_train, y_test = train_test_split(X_s, y,
                                                    shuffle=True, test_size=0.3, random_state=42)


# In[80]:


from sklearn.linear_model import LinearRegression
#Vanilla Linear Regression 

linearRegression = LinearRegression().fit(X_train, y_train)
linearRegression_rmse = rmse(y_test, linearRegression.predict(X_test))
linearRegression_R2 = r2_score(y_test, linearRegression.predict(X_test)) 
print(linearRegression_rmse)
print(linearRegression_R2)


# In[82]:


fig = plt.figure(figsize=(12,8))
ax = plt.axes()

ax.plot(y_test, linearRegression.predict(X_test), 
         marker='o', ls='', ms=3.0)

lim = (0, y_test.max())

ax.set(xlabel='Actual Selling Price', 
       ylabel='Predicted Selling Price', 
       xlim=lim,
       ylim=lim,
       title='Vanilla Model Results');


# In[83]:


from sklearn.linear_model import LassoCV
alphas = np.geomspace(0.1, 400, 1000)
lassoCV = LassoCV(alphas=alphas,
                  max_iter=10000,
                  cv=3).fit(X_train, y_train)

lassoCV_rmse = rmse(y_test, lassoCV.predict(X_test))
lassoCV_R2 = r2_score(y_test, lassoCV.predict(X_test)) 

print(lassoCV.alpha_, lassoCV_rmse)  # Lasso is slower
print(lassoCV_R2)


# In[90]:


fig = plt.figure(figsize=(12,8))
ax = plt.axes()

ax.plot(y_test, lassoCV.predict(X_test), 
         marker='o', ls='', color='#eb631e', ms=3.0)

lim = (0, y_test.max())

ax.set(xlabel='Actual Sales Price', 
       ylabel='Predicted Sales Price', 
       xlim=lim,
       ylim=lim,
       title='Lasso Model Results');


# In[85]:


from sklearn.linear_model import RidgeCV

#Ridge regression 

alphas = np.geomspace(0.01, 20, 1000).tolist()
ridgeCV = RidgeCV(alphas=alphas, cv=3).fit(X_train, y_train)

ridgeCV_rmse = rmse(y_test, ridgeCV.predict(X_test))
ridgeCV_R2 = r2_score(y_test, ridgeCV.predict(X_test)) 

print(ridgeCV.alpha_, ridgeCV_rmse)
print(ridgeCV_R2)


# In[94]:


fig = plt.figure(figsize=(12,8))
ax = plt.axes()

ax.plot(y_test, ridgeCV.predict(X_test), 
         marker='o', ls='', color='g', ms=3.0)

lim = (0, y_test.max())

ax.set(xlabel='Actual Price Sell', 
       ylabel='Predicted Price Sell', 
       xlim=lim,
       ylim=lim,
       title='Ridge Model Results');


# In[92]:


from sklearn.linear_model import ElasticNetCV
alphas = np.geomspace(0.001, 1, 100)
l1_ratios = np.linspace(0.1, 0.9, 10)

elasticNetCV = ElasticNetCV(alphas=alphas, 
                            l1_ratio=l1_ratios,
                            max_iter=1e4).fit(X_train, y_train)

elasticNetCV_rmse = rmse(y_test, elasticNetCV.predict(X_test))
elasticNetCV_R2 = r2_score(y_test, elasticNetCV.predict(X_test)) 

print(elasticNetCV.alpha_, elasticNetCV.l1_ratio_, elasticNetCV_rmse)
print(ridgeCV_R2)


# In[95]:


fig = plt.figure(figsize=(12,8))
ax = plt.axes()

ax.plot(y_test, elasticNetCV.predict(X_test), 
         marker='o', ls='', color='#c965c8', ms=3.0)

lim = (0, y_test.max())

ax.set(xlabel='Actual Price Sell', 
       ylabel='Predicted Price Sell', 
       xlim=lim,
       ylim=lim,
       title='Elastic Net Model Results');


# In[96]:


rmse_vals = [linearRegression_rmse, lassoCV_rmse, ridgeCV_rmse, elasticNetCV_rmse]
R2_vals = [linearRegression_R2, lassoCV_R2, ridgeCV_R2, elasticNetCV_R2] 

labels = ['Linear', 'Lasso', 'Ridge', 'ElasticNet']
metric_df = pd.Series(rmse_vals, index=labels).to_frame()
metric_df.rename(columns={0: 'RMSE'}, inplace=1)
metric_df['R2'] = R2_vals
metric_df


# In[99]:


# Import SGDRegressor and prepare the parameters

from sklearn.linear_model import SGDRegressor

model_parameters = {
    'Linear': {'penalty': 'none'},
    'Lasso': {'penalty': 'l1',
           'alpha': lassoCV.alpha_},
    'Ridge': {'penalty': 'none',
           'alpha': ridgeCV_rmse},
    'ElasticNet': {'penalty': 'elasticnet', 
                   'alpha': elasticNetCV.alpha_,
                   'l1_ratio': elasticNetCV.l1_ratio_}
}

final_rmses = {}
final_R2 = {}
for label, parameters in model_parameters.items():
    # pass arguments 
    SGD = SGDRegressor(**parameters)
    SGD.fit(X_train, y_train)
    final_rmses[label] = rmse(y_test, SGD.predict(X_test))
    final_R2[label] = r2_score(y_test, SGD.predict(X_test))
    
metric_df['RMSE-SGD'] = pd.Series(final_rmses)
metric_df['R2-SGD'] = pd.Series(final_R2)
metric_df

