import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from sklearn.preprocessing import OneHotEncoder
from ml_logic.data import split_data, separate_feature_target



# # Impute then scale numerical values:
# num_transformer = Pipeline([
#     ('imputer', SimpleImputer(strategy="mean")),
#     ('standard_scaler', StandardScaler())
# ])

# # Encode categorical values
# cat_transformer = OneHotEncoder(handle_unknown='ignore')

# # Parallelize "num_transformer" and "cat_transfomer"
# preprocessor = ColumnTransformer([
#     ('num_transformer', num_transformer, ['age', 'bmi']),
#     ('cat_transformer', cat_transformer, ['smoker', 'region'])
# ])


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

def rebalancing_SMOTE (X_train, y_train):

    X_resampled_SMOTE, y_resampled_SMOTE = SMOTE().fit_resample(X_train, y_train)

    return X_resampled_SMOTE, y_resampled_SMOTE

def rebalancing_ADASYN (X_train, y_train):

    X_resampled_ADASYN, y_resampled_ADASYN = ADASYN().fit_resample(X_train, y_train)
    X_resampled_ADASYN['isFraud'] = y_resampled_ADASYN


    return X_resampled_ADASYN, y_resampled_ADASYN
