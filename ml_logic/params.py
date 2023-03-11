"""
the_laundromat package params
load and validate the environment variables in the `.env`
"""

import os
import numpy as np

DATASET_SIZE = "200K"             # ["200k", "500k"]
LOCAL_DATA_PATH = os.path.join(os.path.expanduser('~'), ".AlcaRmsp", "the_laundromat", "raw_data", "data")

# LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'), ".AlcaRmsp", "the_laundromat", "training_outputs")
"""function to persist trained models, params and metrics"""




################## VALIDATIONS #################

env_valid_options = dict(
    DATASET_SIZE=["1k", "10k", "100k", "500k", "50M", "new"])


    # DATA_SOURCE=["local", "big query"],
    # MODEL_TARGET=["local", "gcs", "mlflow"])
