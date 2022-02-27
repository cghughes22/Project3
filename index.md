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
