from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

models = {
    'SVM': SVC(probability=True),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Logistic Regression': LogisticRegression(max_iter=200),
    'GradientBoostingClassifier': GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
                                                             max_depth=1, random_state=0),
    'AdaBoost': AdaBoostClassifier(),
    'KNN': KNeighborsClassifier()
}

def preprocess_data(data, target, labels):
    dataset = None
    for label in labels:
        if labels[label]['use'] == False:
            print('Not Use --> ', label)
        else:
            column = data[label].copy()
            if labels[label]['type'] == 1:
                column.fillna(column.median(), inplace=True)
                mean = column.mean()
                print('mean = ', mean)
                column = column / mean
            else:
                column.fillna(column.mode().values[0], inplace=True)
                categ_list = {category: i for i, category in enumerate(data[label].unique().tolist())}
                column = column.replace(categ_list)
                print(categ_list)
            if dataset is not None:
                dataset = np.vstack([dataset, column.to_numpy()])
            else:
                dataset = np.array([[i for i in column.to_numpy()]])
    dataset = np.transpose(dataset)
    print(dataset.shape)
    X_train, X_test, y_train, y_test = train_test_split(dataset, target, test_size=0.15, random_state=401)
    return X_train, y_train, X_test, y_test


def trainer(X_train, y_train, X_test, y_test, model_name):
    model = models[model_name]
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    cm_model = confusion_matrix(y_test, pred)
    train_accuracy = accuracy_score(y_train, model.predict(X_train))
    test_accuracy = accuracy_score(y_test, pred)
    table_accuracy = classification_report(y_test, pred)
    print('train == ', train_accuracy)
    print('test == ', test_accuracy)

    y_scores = model.predict_proba(X_train)
    y_onehot = pd.get_dummies(y_train, columns=model.classes_)

    return cm_model, test_accuracy, train_accuracy, y_onehot, y_scores, table_accuracy

