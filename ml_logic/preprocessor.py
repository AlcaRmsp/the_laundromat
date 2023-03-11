from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer

from ml_logic.encoders import (transaction_type_encoder, names_encoder)

import numpy as np
import pandas as pd

def preprocess_features(X: pd.DataFrame) -> np.ndarray:


# TRANSACTION PIPE

    transaction_pipeline = make_pipeline([
        ('name_encoder', LabelEncoder)
    transaction_pipeline.fit_transform(df[['nameOrig']])
    transaction_pipeline.fit_transform(df[['nameDest']])

# ACCOUNT NAME PIPE

    name_pipeline = make_pipeline([
        ('transaction_type_encoder', OneHotEncoder)
    name_pipeline.fit_transform(df[['type']])
    name_pipeline.fit_transform(df[['type']])



# time_pipe = make_pipeline(
#             FunctionTransformer(transform_time_features),
#             make_column_transformer(
#                 (OneHotEncoder(
#                     categories=time_categories,
#                     sparse=False,
#                     handle_unknown="ignore"), [2,3]), # correspond to columns ["day of week", "month"], not the others columns
#                 (FunctionTransformer(lambda year: (year-year_min)/(year_max-year_min)), [4]), # min-max scale the columns 4 ["year"]
#                 remainder="passthrough" # keep hour_sin and hour_cos
#                 )
