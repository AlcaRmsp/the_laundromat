
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import recall_score, accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def new_x_train_y_train(data_new):
    X_sampled_train = data_new.drop('isFraud', axis = 1)
    y_sampled_train = data_new['isFraud']

    return X_sampled_train, y_sampled_train

def modellingRFC(X_sampled_train, y_sampled_train, X_test, y_test):
    #state the model
    model_RFC = RandomForestClassifier()

    #instantiate the model
    model_RFC.fit(X_sampled_train, y_sampled_train)

    #Predict the y
    y_pred_RFC = model_RFC.predict(X_test)

    #Scores
    recall_RFC = recall_score(y_test, y_pred_RFC)
    accuracy_RFC = accuracy_score(y_test, y_pred_RFC)

    #Confusion matrix
    cm_RFC = confusion_matrix(y_test, y_pred_RFC)
    confusion_matrix_RFC = pd.DataFrame(data=cm_RFC, columns=['Actual Positive:1', 'Actual Negative:0'],
                                    index=['Predict Positive:1', 'Predict Negative:0'])
    fig, ax = plt.subplots(figsize=(5,5))
    sns.heatmap(confusion_matrix_RFC, annot=True, fmt='d', cmap = 'PuBu', ax = ax)
    plt.savefig('Confusion matrix for RFC model')
    return recall_RFC, accuracy_RFC

def modellingXGBC(X_sampled_train, y_sampled_train, X_test, y_test):
    #state the model
    model_XGBC = XGBClassifier()

    #instantiate the model
    model_XGBC.fit(X_sampled_train, y_sampled_train)

    #Predict the y
    y_pred_XGBC = model_XGBC.predict(X_test)

    #Scores
    recall_XGBC = recall_score(y_test, y_pred_XGBC)
    accuracy_XGBC =accuracy_score(y_test, y_pred_XGBC)

    #confusion matrix
    cm_XGBC = confusion_matrix(y_test, y_pred_XGBC)
    confusion_matrix = pd.DataFrame(data=cm_XGBC, columns=['Actual Positive:1', 'Actual Negative:0'],
                                 index=['Predict Positive:1', 'Predict Negative:0'])
    fig, ax = plt.subplots(figsize=(5,5))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap = 'PuBu', ax = ax)

    return recall_XGBC, accuracy_XGBC
