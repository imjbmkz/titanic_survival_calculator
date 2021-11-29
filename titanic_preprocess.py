## Import packages
from random import seed, randint
from math import isnan
import pandas as pd

## Read data from CSV and make it a pandas dataframe
titanic = pd.read_csv('https://raw.githubusercontent.com/imjbmkz/titanic_survival_calculator/main/titanic.csv')

## Drop samples with missing `survived` (response)
titanic.dropna(axis=0, subset=['survived'], inplace=True)

## Drop unnecessary columns; see README
cols_to_drop = ['pclass', 'name', 'ticket', 'cabin', 
                'embarked', 'boat', 'body', 'home.dest']
titanic.drop(labels=cols_to_drop, axis=1, inplace=True)

## Fill missing fare with median value
titanic.fare.fillna(titanic.fare.median(), inplace=True)

## Recode `sex`: female=1, male=0
titanic.sex.replace({'female':1, 'male':0}, inplace=True)

## Define lambda function that imputes missing values
## with a random number between min(age) and max(age)
min_age = round(titanic.age.min())
max_age = round(titanic.age.max())
imputer = lambda x: randint(min_age, max_age) if isnan(x) else x

## Impute values; set seed for reproducibility
seed(14344)
titanic.age = titanic.age.apply(imputer)

## Update data types
cols_to_int = ['survived', 'sex', 'sibsp', 'parch']
for col in cols_to_int:
  titanic[col] = titanic[col].astype('int32')

## Export the cleaned titanic dataset to csv
titanic.to_csv('titanic_cleaned.csv', index=False)
