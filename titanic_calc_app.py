## Import packages reading data, training models, app dev
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier

## Read the cleaned titanic dataset from csv, split features and response
titanic = pd.read_csv('https://raw.githubusercontent.com/imjbmkz/titanic_survival_calculator/main/titanic_cleaned.csv')
x = titanic.drop(labels='survived', axis=1)
y = titanic.survived

## Instantiate models with the best parameters; derived from titanic_models.py
logit = LogisticRegression(C=10, solver='newton-cg')
tree = DecisionTreeClassifier(max_depth=5)
knn = KNeighborsClassifier(metric='manhattan', n_neighbors=7, weights='uniform')
svc = SVC(C=50, kernel='rbf', probability=True)
forest = RandomForestClassifier(random_state=14344, max_features='sqrt', n_estimators=1000)
xgb = XGBClassifier(learning_rate=0.01, max_depth=3, n_estimators=1000, subsample=0.5)

## Fit the data to all models
models = [logit, tree, knn, svc, forest, xgb]
for model in models:
    model.fit(x, y)

## Streamlit app form
with st.form('titanic_form'):
    st.write('''
    ## Titanic Survival Calculator
    #### What are the chances that ***you*** will survive?
    ''')

    st.write('Fill out all fields of this form and click `calculate` button to calculate survival likelihood.')
    sex = st.selectbox('Sex', options=['male', 'female'])
    age = st.slider('Age', min_value=0, max_value=80, value=25, step=1)
    sibsp  = st.slider('# of sibplings/spouse you\'ll bring:', min_value=0, max_value=8, value=0, step=1)
    parch = st.slider('# of parents/children you\'ll bring:', min_value=0, max_value=9, value=0, step=1)
    fare = st.slider('How much will you pay for (just for yourself in US$)?', min_value=0, max_value=512, value=33, step=1)

    ## Submit button; calculate results
    submitted = st.form_submit_button("Calculate")
    if submitted:

        ## Create dataframe of user inputs
        user_inputs = pd.DataFrame({
            'sex':1 if sex=='female' else 0,
            'age':age,
            'sibsp':sibsp,
            'parch':parch,
            'fare':fare
            }, index=[0])
        
        ## Predict the probabilities of surviving based from inputs
        st.write('Predicted probabilities and survival decisions from 6 optimized models:')

        probabilities = []
        decisions = []
        counter = 0
        for model in models:
            probabilities.append(model.predict_proba(user_inputs)[0, 1])
            decisions.append(model.predict(user_inputs)[0])
            st.write(type(model).__name__, ': Probability: {:.4f}\nDecision: {}\n'.
                     format(probabilities[counter], 'you will survive' if decisions[counter]==1 else 'you will not survive'))
            counter += 1
        
        mean_prob = sum(probabilities) / 6
        majority = max(set(decisions), key=decisions.count)
        
        ## Get the average probability of all models, and the ensemble of their predictions
        st.write('##### Average probability from all 6 models: {:.4f}'.format(mean_prob))
        st.write('##### Majority of the models voted that you {} survive the Titanic mishap.'.format('will' if majority==1 else 'will not'))
