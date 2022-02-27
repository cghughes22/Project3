## Multivariate Regression Analysis and Gradient Boosting

### Multivariate Regression
In order to perform multivariate regression, we need to take a dataset and only use the features that we find to be important for our model. 
### Gradient Boosting
Gradient boosting utilizes a set of decision trees, each with a depth larger than 1, to predict a target label. Usually, these trees have their maximum number of leaves set between 8 and 32. 
### Extreme Gradient Boosting (xgboost)
Unlike gradient boosting, **xgboost** uses regularization parameters to help prevent overfitting of the data. 

### Boston Housing Prices Dataset
The Python code below shows how I set up the dataset's features and its target variable, **cmedv**, which marks the median house price. The features include variables for the average number of rooms in a house (**rooms**), the crime rate (**crime**), nitric oxide concentration in parts per 10 million (**nox**), the pupil-teacher ratio by town (**ptratio**), and the full-value property tax rate per $10,000 (**tax**).
```Python
df = pd.read_csv('drive/MyDrive/DATA 410/Boston Housing Prices.csv')
features = ['crime','rooms','residential','industrial','nox','older','distance','highway','tax','ptratio','lstat']
X = np.array(df[features])
y = np.array(df['cmedv']).reshape(-1,1)
dat = np.concatenate([X,y], axis=1)
```
After I set up X and y, I created training and test datasets, with the latter's size being 30% of the actual data.
```Python
from sklearn.metrics import mean_absolute_error
dat_train, dat_test = tts(dat, test_size=0.3, random_state=1234)
```

### Multivariate Regression on Boston Housing Prices Dataset
I started by setting up a linear regression model
```Python
lm = LinearRegression()
lm.fit(dat_train[:,:-1],dat_train[:,-1])
yhat_lm = lm.predict(dat_test[:,:-1])
mae_lm = mean_absolute_error(dat_test[:,-1], yhat_lm)
print("MAE Linear Model = ${:,.2f}".format(1000*mae_lm))
```
The mean absolute error of this model comes out to be $3,640.02. I then used similar code to set up the Ridge and Lasso regressions, and the obtained values for MAE are $3,600.77 and $3,619.90, respectively. 
Afterwards, I scaled the data and repeated those Ridge and Lasso regressions, with the values for mean absolute error coming out to be $3,443.23 and $3,499.16, respectively. The code for the scaled Ridge regression is shown below.

```Python
scale = StandardScaler()
lr = Ridge(alpha=45)
lr.fit(scale.fit_transform(dat_train[:,:-1]),dat_train[:,-1])
yhat_lr = lr.predict(scale.transform(dat_test[:,:-1]))
mae_lr = mean_absolute_error(dat_test[:,-1], yhat_lr)
print("MAE Ridge Model = ${:,.2f}".format(1000*mae_lr))
```

### XGBoost on Boston Housing Prices Dataset
```Python

```

### Cars Dataset

```Python
df = pd.read_csv('drive/MyDrive/DATA 410/cars.csv')
features = ['CYL','ENG','WGT']
X = np.array(df[features])
y = np.array(df['MPG']).reshape(-1,1)
dat = np.concatenate([X,y], axis=1)
```

```Python
from sklearn.metrics import mean_absolute_error
dat_train, dat_test = tts(dat, test_size=0.25, random_state=1234)
```

### Multivariate Regression on Cars Dataset

```Python
lm = LinearRegression()
lm.fit(dat_train[:,:-1],dat_train[:,-1])
yhat_lm = lm.predict(dat_test[:,:-1])
mae_lm = mean_absolute_error(dat_test[:,-1], yhat_lm)
print("MAE Linear Model = ${:,.2f}".format(1000*mae_lm))
```
I then ran repeated the process of running Ridge and Lasso regressions on the testing and training datasets, in which the values for mean absolute error are $3,477.03 and $3,474.79, respectively. Like with the Boston Housing Prices dataset, I also scaled the Cars data and ran the same Ridge and Lasso regressions, obtaining MAE values of $3,516.10 and $3,466.36, respectively. Displayed below is the code for the scaled Lasso regression of the Cars data.
```Python
scale = StandardScaler()
ls = Lasso(alpha=0.15)
ls.fit(scale.fit_transform(dat_train[:,:-1]),dat_train[:,-1])
yhat_ls = ls.predict(scale.transform(dat_test[:,:-1]))
mae_ls = mean_absolute_error(dat_test[:,-1], yhat_ls)
print("MAE Lasso Model = ${:,.2f}".format(1000*mae_ls))
```

### XGBoost on Cars Dataset

```Python

```
