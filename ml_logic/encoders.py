import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from ml_logic.data import split_data, separate_feature_target


def transaction_type_encoder(df: pd.DataFrame) -> pd.DataFrame:
    '''
    function encoding the kind of transactions 'CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER'
    '''
    # Instantiate the OneHotEncoder
    ohe_binary = OneHotEncoder(sparse = False, drop="if_binary")

    # Fit encoder
    ohe_binary.fit(df[['type']])

    # Transform the current "Street" column
    df[ohe_binary.get_feature_names_out()] = ohe_binary.transform(df[['type']])

    # Drop the column "Street" which has been encoded
    df.drop(columns = ["type"], inplace = True)

    return df


def names_encoder(df: pd.DataFrame) -> pd.DataFrame:
    ''' function encoding the originator and destinator names'''

    # create a label encoder object
    le = LabelEncoder()

    # apply the label encoder to non-numeric columns
    df['nameOrig'] = le.fit_transform(df['nameOrig'])
    df['nameDest'] = le.fit_transform(df['nameDest'])

    return df
