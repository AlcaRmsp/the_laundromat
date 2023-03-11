
import os
import pandas as pd
from sklearn.model_selection import train_test_split


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    clean raw data by removing irrelevant columns
    and renaming columns
    """

    df = df.drop(columns=['isFlaggedFraud'])

    df = df.rename(columns={'oldbalanceOrg':'oldBalanceOrig', 'newbalanceOrig':'newBalanceOrig', \
                        'oldbalanceDest':'oldBalanceDest', 'newbalanceDest':'newBalanceDest'})
    df['errorBalanceOrig']=df['newBalanceOrig'] + df['amount'] - df['oldBalanceOrig']
    df['errorBalanceDest']=df['newBalanceDest'] + df['amount'] - df['oldBalanceDest']

    return df

def separate_feature_target (df: pd.DataFrame):

    """ Separate the features and target variables """

    X = df.drop('isFraud', axis=1)
    y = df['isFraud']

    return X, y

def split_data (X:pd.DataFrame, y:pd.Series):

    """Split the data between train and test sets"""

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def create_new_features (df:pd.DataFrame):

    """Create 'balance error in originator account' feature and 'balance error in destination account feature """

    df['errorBalanceOrig']=df['newBalanceOrig'] + df['amount'] - df['oldBalanceOrig']
    df['errorBalanceDest']=df['newBalanceDest'] + df['amount'] - df['oldBalanceDest']

    return df
