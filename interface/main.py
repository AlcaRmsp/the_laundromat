import numpy as np
import pandas as pd
import os

# import functions from other files

from ml_logic.data import clean_data, create_new_features, separate_feature_target, split_data, rebalancing_SMOTE, resample
from ml_logic.params import DATA_SOURCE
from ml_logic.encoders import transaction_type_encoder, names_encoder
from ml_logic.model import modellingLR, modellingDTC, modellingRFC, modellingXGBC

data_raw_path = os.path.join(DATA_SOURCE)

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
    df_processed = resample(df_resampled_SMOTE)

    return df_processed

    #Create new X_processed and y_processed

def new_x_train_y_train(data_new):
    X_sampled_train = data_new.drop('isFraud', axis = 1)
    y_sampled_train = data_new['isFraud']

    return X_sampled_train, y_sampled_train


def train (X_sampled_train, y_sampled_train, X_test, y_test, model):

    if model is 'Logistic Regression':
        model = modellingLR(X_sampled_train, y_sampled_train, X_test, y_test)

    if model is 'Decision Tree Classifier':
        model = modellingDTC (X_sampled_train, y_sampled_train, X_test, y_test)

    if model is "Random Forest Classifier":
        model = modellingRFC (X_sampled_train, y_sampled_train, X_test, y_test)

 call funtion from model.py

 return model, confusion matrix



    # # Save trained model
    # params = dict(
    #     learning_rate=learning_rate,
    #     batch_size=batch_size,
    #     patience=patience
    # )

    # save_model(model, params=params, metrics=metrics)

    # # 🧪 Write outputs so that they can be tested by make test_train_at_scale (do not remove)
    # write_result(name="test_preprocess_and_train", subdir="train_at_scale", metrics=metrics)

    # print("✅ preprocess_and_train() done")




# def train():
#     """
#     Train on the full (already preprocessed) dataset, by loading it
#     chunk-by-chunk, and updating the weight of the model for each chunk.
#     Save model, compute validation metrics on a holdout validation set that is
#     common to all chunks.
#     """
#     print("\n ⭐️ Use case: train")

#     # Validation set: load a validation set common to all chunks and create X_val, y_val
#     data_val_processed_path = os.path.abspath(os.path.join(
#         LOCAL_DATA_PATH, "processed", f"val_processed_{VALIDATION_DATASET_SIZE}.csv"
#     ))

#     data_val_processed = pd.read_csv(
#         data_val_processed_path,
#         skiprows= 1, # skip header
#         header=None,
#         dtype=DTYPES_PROCESSED_OPTIMIZED
#     ).to_numpy()

#     X_val = data_val_processed[:, :-1]
#     y_val = data_val_processed[:, -1]

#     # Iterate over the full training dataset in chunks.
#     # Break out of the loop if you receive no more data to train upon!
#     model = None
#     chunk_id = 0
#     metrics_val_list = []  # store each metrics_val_chunk

#     while (True):
#         print(f"Loading and training on preprocessed chunk n°{chunk_id}")

#         # Load chunk of preprocess data and create (X_train_chunk, y_train_chunk)
#         path = os.path.abspath(os.path.join(
#             LOCAL_DATA_PATH, "processed", f"train_processed_{DATASET_SIZE}.csv"))

#         try:
#             data_processed_chunk = pd.read_csv(
#                 path,
#                 skiprows=(chunk_id * CHUNK_SIZE) + 1, # skip header
#                 header=None,
#                 nrows=CHUNK_SIZE,
#                 dtype=DTYPES_PROCESSED_OPTIMIZED,
#             ).to_numpy()

#         except pd.errors.EmptyDataError:
#             data_processed_chunk = None  # end of data

#         # Break out of while loop if we have no data to train upon
#         if data_processed_chunk is None:
#             break

#         X_train_chunk = data_processed_chunk[:, :-1]
#         y_train_chunk = data_processed_chunk[:, -1]

#         learning_rate = 0.001
#         batch_size = 256
#         patience=2

#         # Train a model *incrementally*, and store the val MAE of each chunk in `metrics_val_list`
#         # $CODE_BEGIN
#         if model is None:
#             model = initialize_model(X_train_chunk)

#         model = compile_model(model, learning_rate)

#         model, history = train_model(
#             model,
#             X_train_chunk,
#             y_train_chunk,
#             batch_size=batch_size,
#             patience=patience,
#             validation_data=(X_val, y_val)
#         )

#         metrics_val_chunk = np.min(history.history['val_mae'])
#         metrics_val_list.append(metrics_val_chunk)

#         print(metrics_val_chunk)
#         # $CODE_END

#         chunk_id += 1

#     # Return the last value of the validation MAE
#     val_mae = metrics_val_list[-1]

#     # Save model and training params
#     params = dict(
#         learning_rate=learning_rate,
#         batch_size=batch_size,
#         patience=patience,
#         incremental=True,
#         chunk_size=CHUNK_SIZE
#     )

#     print(f"\n✅ Trained with MAE: {round(val_mae, 2)}")

#     save_model(model=model, params=params, metrics=dict(mae=val_mae))

#     print("✅ Model trained and saved")

# # $ERASE_END

# def pred(X_pred: pd.DataFrame = None) -> np.ndarray:

#     if X_pred is None:
#         X_pred = pd.DataFrame(dict(
#             key=["2013-07-06 17:18:00"],  # useless but the pipeline requires it
#             pickup_datetime=["2013-07-06 17:18:00 UTC"],
#             pickup_longitude=[-73.950655],
#             pickup_latitude=[40.783282],
#             dropoff_longitude=[-73.984365],
#             dropoff_latitude=[40.769802],
#             passenger_count=[1]
#         ))

#     model = load_model()

#     # Preprocess the new data
#     # $CODE_BEGIN
#     X_processed = preprocess_features(X_pred)
#     # $CODE_END

#     # Make a prediction
#     # $CODE_BEGIN
#     y_pred = model.predict(X_processed)
#     # $CODE_END

#     # 🧪 Write outputs so that they can be tested by make test_train_at_scale (do not remove)
#     write_result(name="test_pred", subdir="train_at_scale", y_pred=y_pred)
#     print("✅ prediction done: ", y_pred, y_pred.shape)

#     return y_pred


if __name__ == '__main__':
    try:
        preprocess_and_train()
        pred()
    except:
        import ipdb, traceback, sys
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)
