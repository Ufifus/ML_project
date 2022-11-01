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
    print(table_accuracy.splitlines())
    classification_matrix = create_classification_report(table_accuracy, y_test, pred)
    print('train == ', train_accuracy)
    print('test == ', test_accuracy)
    print('Class matrix == ', classification_matrix)

    y_scores = model.predict_proba(X_test)
    y_onehot = pd.get_dummies(y_test, columns=model.classes_)

    return cm_model, test_accuracy, train_accuracy, y_onehot, y_scores, classification_matrix


def create_classification_report(table_accuracy, y_test, pred):
    """Create matrix with columns:
    'Sensitivity' --> SE = TP/(TP+FN)
    'Specificity' --> SP = FP/(FP+TN)
    'PPV' --> PPV = TP/(TP+FP)
    'NNV' --> NPV = TN/(TN+FN)
    """
    uniq_vals = list(set(y_test))
    dict_results = {k: {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0} for k in uniq_vals}
    for label in uniq_vals:
        for expect, fact in zip(y_test, pred):
            if label == expect and expect == fact:
                dict_results[label]['TP'] += 1
            elif label != expect and label != fact:
                dict_results[label]['TN'] += 1
            elif label != expect and label == fact:
                dict_results[label]['FP'] += 1
            elif label == expect and label != fact:
                dict_results[label]['FN'] += 1
    print(dict_results)

    table_strings = table_accuracy.splitlines()
    table_strings = [[el for el in s.split(' ') if el != ''] for s in table_strings if s != '']
    table_strings[0] = ['label'] + table_strings[0] + ['SE', 'SP', 'PPV', 'NPV']
    classification_matrix = {}
    for i, (k, v) in enumerate(dict_results.items()):
        print(k, v)
        try:
            SE = v['TP'] / (v['TP'] + v['FN'])
        except ZeroDivisionError:
            SE = 0
        try:
            SP = v['FP'] / (v['FP'] + v['TN'])
        except ZeroDivisionError:
            SP = 0
        try:
            PPV = v['TP'] / (v['TP'] + v['FP'])
        except ZeroDivisionError:
            PPV = 0
        try:
            NPV = v['TN'] / (v['TN'] + v['FN'])
        except ZeroDivisionError:
            NPV = 0

        classification_matrix[k] = [round(SE, 2), round(SP, 2), round(PPV, 2), round(NPV, 2)]
        table_strings[i+1] =table_strings[i+1] + classification_matrix[k]

    print(table_strings)
    classification_matrix = pd.DataFrame(table_strings[1:i+2], columns=table_strings[0])

    print(classification_matrix)
    return classification_matrix