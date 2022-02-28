## Multivariate Regression Analysis and Gradient Boosting

### Multivariate Regression
In order to perform multivariate regression, we need to take a dataset and only use the features that we find to be important for our model. 
### Gradient Boosting
Gradient boosting utilizes a set of decision trees, each with a depth larger than 1, to predict a target label. Usually, these trees have their maximum number of leaves set between 8 and 32. We start off by calculating the average value of the target variable. Afterwards, we calculate the residuals by subtracting the predicted values from the actual values, constructing a decision tree with residual predictions in the process. We then predict the actual target variable using the values found in the decision tree, and compute new residuals. We repeat this process until reaching the value of the hyperparameter, and make a final prediction for the value of the target variable.
### Extreme Gradient Boosting (xgboost)
Unlike gradient boosting, **xgboost** uses regularization parameters to help prevent overfitting of the data. We start with an arbitrary initial prediction, which could be the average of a given variable. Then, for every sample, we calculate the residuals by subtracting the predicted value from each actual value. We then use a linear scan by selecting thresholds between sets of points in order to decide the best split among a given feature variable. This helps us construct a decision tree, after which we start to calculate gain, the improvement in accuracy brought upon by the split. Gain is then calculated for every split, and thus every leaf of the tree, and use the sign of each split's gain to determine whether or not the split yields better results than if we left the tree alone. If it's positive, we can keep splitting further down the tree, but if it's negative, then we move onto the next split over. After all necessary gain calculations, we start making our predictions by adding the initial prediction and the product of the learning rate and the decision tree's individual predictions, creating new residuals in the process. We repeat this until we reach the maximum number of estimators.

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
I then conducted tests to find the optimal values for alpha and mean absolute error for the Ridge and Lasso regressions, and in doing so, I created scatterplot for each regression, both of which are shown below. The optimal alpha values for the respective regressions are 0.43 and 0.13, and the optimal values for mean absolute error are $3,442.81 and $3,489.26.
<img src="Assets/Housing Price Ridge Scatterplot.png" width="800" height="600" alt=hi class="inline"/>

<img src="Assets/Housing Price Lasso Scatterplot.png" width="800" height="600" alt=hi class="inline"/>

### XGBoost on Boston Housing Prices Dataset

```Python
mse_blwr = []
mse_xgb = []
for i in range(5):
  kf = KFold(n_splits=10,shuffle=True,random_state=i)
  for idxtrain, idxtest in kf.split(X):
    xtrain = X[idxtrain]
    ytrain = y[idxtrain]
    ytest = y[idxtest]
    xtest = X[idxtest]
    xtrain = scale.fit_transform(xtrain)
    xtest = scale.transform(xtest)
    dat_train = np.concatenate([xtrain,ytrain.reshape(-1,1)],axis=1)
    dat_test = np.concatenate([xtest,ytest.reshape(-1,1)],axis=1)
    model_xgb = xgb.XGBRegressor(objective ='reg:squarederror',n_estimators=100,reg_lambda=20,alpha=1,gamma=10,max_depth=1)
    model_xgb.fit(xtrain,ytrain)
    yhat_xgb = model_xgb.predict(xtest)
    mse_xgb.append(mse(ytest,yhat_xgb))
```

### Cars Dataset

```Python
df = pd.read_csv('drive/MyDrive/DATA 410/cars.csv')
features = ['CYL','ENG','WGT']
X = np.array(df[features])
y = np.array(df['MPG']).reshape(-1,1)
dat = np.concatenate([X,y], axis=1)
```
Similarly to the Boston dataset, I created training and testing sets, this time decreasing the size of the testing set to 25% of the original data, using the following code.
```Python
from sklearn.metrics import mean_absolute_error
dat_train, dat_test = tts(dat, test_size=0.25, random_state=1234)
```

### Multivariate Regression on Cars Dataset
Like with the previous dataset, I started out by running a typical linear regression, obtaining a mean absolute error of 3,477.77.
```Python
lm = LinearRegression()
lm.fit(dat_train[:,:-1],dat_train[:,-1])
yhat_lm = lm.predict(dat_test[:,:-1])
mae_lm = mean_absolute_error(dat_test[:,-1], yhat_lm)
print("MAE Linear Model = ${:,.2f}".format(1000*mae_lm))
```
I then ran repeated the process of running Ridge and Lasso regressions on the testing and training datasets, in which the values for mean absolute error are 3,477.03 and 3,474.79, respectively. Like with the Boston Housing Prices dataset, I also scaled the Cars data and ran the same Ridge and Lasso regressions, obtaining MAE values of 3,516.10 and 3,466.36, respectively. Displayed below is the code for the scaled Lasso regression of the Cars data.
```Python
scale = StandardScaler()
ls = Lasso(alpha=0.15)
ls.fit(scale.fit_transform(dat_train[:,:-1]),dat_train[:,-1])
yhat_ls = ls.predict(scale.transform(dat_test[:,:-1]))
mae_ls = mean_absolute_error(dat_test[:,-1], yhat_ls)
print("MAE Lasso Model = ${:,.2f}".format(1000*mae_ls))
```
After that, I used the same methods as with the Boston Housing Prices dataset to find optimal values for alpha and mean absolute error under the Ridge and Lasso regression methods. These respective optimal values for alpha are 0.06 and 0.19, and the optimal values for mean absolute error are $3,465.77
<img src="Assets/Cars Ridge Scatterplot.png" width="800" height="600" alt=hi class="inline"/>

<img src="Assets/Cars Lasso Scatterplot.png" width="800" height="600" alt=hi class="inline"/>

### XGBoost on Cars Dataset
Using XGBoost on the dataset, I obtained a cross-validated mean square error value of 16.5594, compared to a slightly higher value of 16.7022 under the boosted LOWESS regression. Below is the code for the function to calculate this value.
```Python

mse_blwr = []
mse_xgb = []
for i in range(5):
  kf = KFold(n_splits=10,shuffle=True,random_state=i)
  for idxtrain, idxtest in kf.split(X):
    xtrain = X[idxtrain]
    ytrain = y[idxtrain]
    ytest = y[idxtest]
    xtest = X[idxtest]
    xtrain = scale.fit_transform(xtrain)
    xtest = scale.transform(xtest)
    dat_train = np.concatenate([xtrain,ytrain.reshape(-1,1)],axis=1)
    dat_test = np.concatenate([xtest,ytest.reshape(-1,1)],axis=1)
    model_xgb = xgb.XGBRegressor(objective ='reg:squarederror',n_estimators=100,reg_lambda=20,alpha=1,gamma=10,max_depth=1)
    model_xgb.fit(xtrain,ytrain)
    yhat_xgb = model_xgb.predict(xtest)
    mse_xgb.append(mse(ytest,yhat_xgb))
```
