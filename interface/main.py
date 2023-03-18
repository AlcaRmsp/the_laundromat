import numpy as np
import pandas as pd
import os
import pickle

# import functions from other files

from ml_logic.data import clean_data, create_new_features, separate_feature_target, split_data, rebalancing_SMOTE, resample
from ml_logic.params import DATA_SOURCE
from ml_logic.encoders import transaction_type_encoder, names_encoder
from ml_logic.model import modellingLR, modellingDTC, modellingRFC, modellingXGBC, predictingLR, predictingDTC, predictingRFC, predictingXGBC

data_raw_path = DATA_SOURCE

def preprocess(data_raw_path):
    """
    Load synthetic dataset in memory, clean and preprocess it"""

    # Retrieve raw data
    df = pd.read_csv(data_raw_path)

    # Clean data using ml_logic.data.clean_data
    clean_df = clean_data(df)
    df_new_features = create_new_features(clean_df)

     #Encode transaction type and account names
    df_transaction_encoded= transaction_type_encoder (df_new_features)
    df_names_encoded = names_encoder (df_transaction_encoded)

    #Split dataset into features and target
    X, y = separate_feature_target(df_names_encoded)

    #Spliting the dataset into train and test
    X_train, X_test, y_train, y_test = split_data(X, y)

    #Oversampling the minority class with SMOTE
    df_resampled_SMOTE = rebalancing_SMOTE(X_train, y_train)

    #Taking a sample of the balanced dataset to train our models
    df_processed = resample(df_resampled_SMOTE, 100000)

    return df_processed

    #Create new X_processed and y_processed

def new_x_train_y_train(data_new):
    X_sampled_train = data_new.drop('isFraud', axis = 1)
    y_sampled_train = data_new['isFraud']

    return X_sampled_train, y_sampled_train


def train (X_sampled_train, y_sampled_train, model):

    """This function will NOT take any user input because  we want to have models pre-trained and then saved"""

    if model == 'Logistic Regression':
        model = modellingLR(X_sampled_train, y_sampled_train)

    if model == 'Decision Tree Classifier':
        model = modellingDTC (X_sampled_train, y_sampled_train)

    if model == "Random Forest Classifier":
        model = modellingRFC (X_sampled_train, y_sampled_train)

    if model == "XGB Classifier":
        model = modellingXGBC (X_sampled_train, y_sampled_train)

    return model


def save_trained_model (model, model_name):
    with open(f"trained_models/{model_name}.pickle", "wb") as handle:
        pickle.dump(model, handle)


def load_trained_model (model_name):
    loaded_model = pickle.load(open(f'trained_models/{model_name}.pickle'))
    return loaded_model

def pred (X_test, y_test, model):
    """ This function will take user input and will use saved model from train function"""
    model = pickle.loads("file to pickle file in")

    if model == 'Logistic Regression':
        prediction = predictingLR(X_test, y_test)

    if model == 'Decision Tree Classifier':
        prediction = predictingDTC (X_test, y_test)

    if model == "Random Forest Classifier":
        prediction = predictingRFC (X_test, y_test)

    if model == "XGB Classifier":
        prediction = predictingXGBC (X_test, y_test)

    return prediction


if __name__ == '__main__':

    """train and save all four models, prediction only will be used for FE"""

    df_processed = preprocess(data_raw_path)
    print(df_processed.head())
    X_sampled_train, y_sampled_train = new_x_train_y_train(df_processed)
    model_list = ['Logistic Regression',
                    'Decision Tree Classifier',
                    "Random Forest Classifier",
                    "XGB Classifier"]
    for model in model_list:
        trained_model = train(X_sampled_train, y_sampled_train, model)
        save_trained_model(trained_model, model)
    # train()
    # save_trained_model()
    # pred()
