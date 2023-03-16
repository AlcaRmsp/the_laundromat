import os

"""Loads dataset from user's local source"""

parent_path = os.path.dirname(os.path.dirname(__file__))
DATA_SOURCE = os.path.join(parent_path, 'raw_data/data.csv')
