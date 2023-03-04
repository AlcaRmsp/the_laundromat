import os
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from ml_logic.data import split_data, separate_feature_target


def transaction_type_encoder(df: pd.DataFrame) -> pd.DataFrame:
    '''
    function encoding the kind of transactions 'CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER'
    '''


    # Instantiate the Ordinal Encoder
    ordinal_encoder = OrdinalEncoder()

    # Fit it
    ordinal_encoder.fit(df[["type"]])


    # Transforming categories into ordered numbers
    df["type"] = ordinal_encoder.transform(df[["type"]])

    return df


def names_encoder(df: pd.DataFrame) -> pd.DataFrame:
    ''' function encoding the originator and destinator names'''

    # create a label encoder object
    le = LabelEncoder()

    # apply the label encoder to non-numeric columns
    df['nameOrig'] = le.fit_transform(df['nameOrig'])
    df['nameDest'] = le.fit_transform(df['nameDest'])

    return df

def rebalancing_SMOTE (X_train, y_train):

    X_resampled_SMOTE, y_resampled_SMOTE = SMOTE().fit_resample(X_train, y_train)

    return X_resampled_SMOTE, y_resampled_SMOTE

def rebalancing_ADASYN (X_train, y_train):

    X_resampled_ADASYN, y_resampled_ADASYN = ADASYN().fit_resample(X_train, y_train)
    X_resampled_ADASYN['isFraud'] = y_resampled_ADASYN


    return X_resampled_ADASYN, y_resampled_ADASYN

