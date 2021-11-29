## Import model classes from sklearn
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier

## Read data from CSV and make it a pandas dataframe
titanic = pd.read_csv('https://raw.githubusercontent.com/imjbmkz/titanic_survival_calculator/main/titanic_cleaned.csv')

## Split features and response variable
x = titanic.drop(labels='survived', axis=1)
y = titanic.survived

## Instantiate models
logit = LogisticRegression()
tree = DecisionTreeClassifier()
knn = KNeighborsClassifier()
svc = SVC()
forest = RandomForestClassifier(random_state=14344)
xgb = XGBClassifier()

## Fit data to the models; print accuracy per model
models = [logit, tree, knn, svc, forest, xgb]
best_acc = ['model', 0]
for model in models:
  model.fit(x, y)
  accuracy = model.score(x, y)
  model_name = type(model).__name__
  print('Accuracy of {}: {:.4f}'.format(model_name, accuracy))
  if accuracy > best_acc[1]:
    best_acc[0] = model_name
    best_acc[1] = accuracy

## Print the model with the best accuracy
model_name = best_acc[0]
accuracy = best_acc[1]
print('\nBest model: {} - Accuracy: {:.4f}\n'.format(model_name, accuracy))
  
## Define repeated stratified cross validation
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=14344)

## Logistic regression hyperparameter tuning
print('Tuning logistic regression parameters')
solvers = ['newton-cg', 'lbfgs', 'liblinear']
c_values = [100, 10, 1.0, 0.1, 0.01]
grid = dict(solver=solvers, C=c_values)

grid_search = GridSearchCV(
    estimator=logit, param_grid=grid, n_jobs=-1, cv=cv, 
    scoring='accuracy', error_score=0)

grid_result = grid_search.fit(x, y)
logit_best = grid_result.best_estimator_
print('Logistic regression best parameters: ', grid_result.best_params_)

## Decision tree hyperparameter tuning
print('\nTuning decision trees parameters')
max_depth = [None, 3, 5, 7, 9, 11]
grid = dict(max_depth=max_depth)

grid_search = GridSearchCV(
    estimator=tree, param_grid=grid, n_jobs=-1, cv=cv, 
    scoring='accuracy', error_score=0)

grid_result = grid_search.fit(x, y)
tree_best = grid_result.best_estimator_
print('Decision trees best parameters: ', grid_result.best_params_)

## KNN hyperparameter tuning
print('\nTuning KNN parameters')
n_neighbors = range(1, 21, 2)
weights = ['uniform', 'distance']
metric = ['euclidean', 'manhattan', 'minkowski']
grid = dict(n_neighbors=n_neighbors, weights=weights, metric=metric)

grid_search = GridSearchCV(
    estimator=knn, param_grid=grid, n_jobs=-1, cv=cv, 
    scoring='accuracy', error_score=0)

grid_result = grid_search.fit(x, y)
knn_best = grid_result.best_estimator_
print('KNN best parameters: ', grid_result.best_params_)

## SVC hyperparameter tuning
print('\nTuning SVC parameters')
kernel = ['poly', 'rbf', 'sigmoid']
C = [50, 10, 1.0, 0.1, 0.01]
grid = dict(kernel=kernel, C=C)

grid_search = GridSearchCV(
    estimator=svc, param_grid=grid, n_jobs=-1, cv=cv, 
    scoring='accuracy', error_score=0)

grid_result = grid_search.fit(x, y)
svc_best = grid_result.best_estimator_
print('SVC best parameters: ', grid_result.best_params_)

## Random forest hyperparameter tuning
print('\nTuning random forest parameters')
n_estimators = [10, 100, 1000]
max_features = ['sqrt', 'log2']
grid = dict(n_estimators=n_estimators, max_features=max_features)

grid_search = GridSearchCV(
    estimator=forest, param_grid=grid, n_jobs=-1, cv=cv, 
    scoring='accuracy', error_score=0)

grid_result = grid_search.fit(x, y)
forest_best = grid_result.best_estimator_
print('Random forest best parameters: ', grid_result.best_params_)

## XGB hyperparameter tuning
print('\nTuning XGB parameters')
n_estimators = [10, 100, 1000]
learning_rate = [0.001, 0.01, 0.1]
subsample = [0.5, 0.7, 1.0]
max_depth = [3, 7, 9]
grid = dict(n_estimators=n_estimators, learning_rate=learning_rate,
            subsample=subsample, max_depth=max_depth)

grid_search = GridSearchCV(
    estimator=xgb, param_grid=grid, n_jobs=-1, cv=cv, 
    scoring='accuracy', error_score=0)

grid_result = grid_search.fit(x, y)
xgb_best = grid_result.best_estimator_
print('XGB best parameters: ', grid_result.best_params_)

## Print accuracy of the models with best parameters
best_models = [logit_best, tree_best, knn_best, 
               svc_best, forest_best, xgb_best]
best_acc = ['model', 0]
for best_model in best_models:
  accuracy = best_model.score(x, y)
  model_name = type(best_model).__name__
  print('Accuracy of', type(best_model).__name__, ': {:.4f}'.
        format(accuracy))
  if accuracy > best_acc[1]:
    best_acc[0] = model_name
    best_acc[1] = accuracy

## Print the model with the best accuracy
model_name = best_acc[0]
accuracy = best_acc[1]
print('Best model: {} - Accuracy: {:.4f}'.format(model_name, accuracy))
