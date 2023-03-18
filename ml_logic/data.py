import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


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

def create_new_features (df:pd.DataFrame):

    """Create 'balance error in originator account' feature and 'balance error in destination account feature """

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

def rebalancing_SMOTE (X_train, y_train):

    """Initial dataset is extremly unbalanced. This function oversamples the minority class ("fraud") to balance the dataset"""

    X_resampled_SMOTE, y_resampled_SMOTE = SMOTE().fit_resample(X_train, y_train)
    X_resampled_SMOTE['isFraud'] = y_resampled_SMOTE
    df_resampled_SMOTE = X_resampled_SMOTE

    return df_resampled_SMOTE


def resample(df_resampled_SMOTE, sample_size: int):

    """Takes a sample of the dataset. Sample size can be chosen by user"""
    fraud = df_resampled_SMOTE[df_resampled_SMOTE.isFraud == 1].sample(sample_size)
    notfraud = df_resampled_SMOTE[df_resampled_SMOTE.isFraud == 0].sample(sample_size)
    data_new = pd.concat([fraud, notfraud], axis=0)
    return data_new
