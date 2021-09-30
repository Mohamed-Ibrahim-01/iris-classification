import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

RANDOM_STATE = [30,40]
LOG_SPACE = np.logspace(start = 0, stop = 10, num = 100, base = 2 , dtype = 'float64') 

classifiers = [
    {
        'name':'Logistic Regression',
        'parameters':{
            'random_state':RANDOM_STATE,
            'C':LOG_SPACE
        },
        'method':LogisticRegression
    },
    {
        'name':'Naive Bayes',
        'parameters':{},
        'method':GaussianNB
    },
    {
        'name':'SVM',
        'parameters':{
            'kernel':['linear', 'poly', 'rbf', 'sigmoid'],
            'C':np.logspace(start = 0, stop = 10, num = 100, base = 2 , dtype = 'float64')
        },
        'method':SVC
    },
    {
        'name':'Decision Tree',
        'parameters':{'max_depth':[1,2,3,4,5]},
        'method': DecisionTreeClassifier
    },
    {
        'name':'Neural Network',
        'parameters':{
            'solver':['lbfgs'],
            'alpha':[1e-5, 1e-3],
            'hidden_layer_sizes':[(3, 3),(5,5),(10,10)],
            'max_iter':[700,750],
            'random_state':RANDOM_STATE
        },
        'method':MLPClassifier
    },
    {
        'name':'Random Forest' ,
        'parameters':{
            'max_depth':[1,2,3,4,5],
            'n_estimators':[10,20,30,40,50,60,70,80,90,100,110,120],
            'max_features':[1,2,3]
        },
        'method':RandomForestClassifier
    }
]
