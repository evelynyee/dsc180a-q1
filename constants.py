"""
Constants for the DSC 180A Quarter 1 project.
"""
import os
import pickle
import json

NYT = 'nyt'
TWENTY_NEWS = '20news'

CONWEA_DATA_PATH = os.path.abspath(os.path.join('..','ConWea','data'))
DATA_PATH = 'data'

def get_data_path(set, granularity='coarse', type='data', local=True):
    """Retrieve path to data file."""
    if local:
        path = DATA_PATH
    else:
        path = CONWEA_DATA_PATH
    path = os.path.join(path, set,granularity)
    if type == 'seedwords':
        path = os.path.join(path, 'seedwords.json')
    elif type == 'data':
        path = os.path.join(path, 'df.pkl')
    else:
        raise ValueError("Invalid argument: `type` must be one of ['data','seedwords']")
    return path

def get_data(set, granularity='coarse', type='data'):
    """Retrieve data file."""
    path = os.path.join(DATA_PATH, set,granularity)
    save_results = None
    data = None
    path = get_data_path(set, granularity=granularity,type=type)
    if not os.path.isfile(path): # file not in the local directory
        save_results = path
        if not os.path.isdir(os.path.split(save_results)[0]): # local directory does not exist
            os.makedirs(os.path.split(save_results)[0])
        path = get_data_path(set, granularity=granularity,type=type, local=False) # get data from conwea repo
    if type == 'seedwords':
        with open(path, 'r') as f:
            data = json.load(f)
        if save_results:
            with open(save_results, 'w') as f:
                json.dump(data,f,indent=4)
    elif type == 'data':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        if save_results:
            with open(save_results, 'wb') as f:
                pickle.dump(data,f)
    return data
