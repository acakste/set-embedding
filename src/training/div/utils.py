# Some function mappings to use the strings in the experiment_config.yaml file as functions.
from src.training.distortion.min_distortion import fn_quadratic
from src.training.distortion.train_to_reference import sample_one

import pandas as pd
import os

function_mappings = {
    "fn_quadratic": fn_quadratic,
    "sample_one": sample_one
}

def select_function(fn_str):
    try:
        return function_mappings[fn_str]
    except KeyError:
        print(f"{fn_str} is an invalid function.")

def get_readymade_names(file="../../data/id_name_mapping_readymades.json"):
    """Takes a list of playlist_id:s and return the associated names, based on local file atm."""
    df = pd.read_json(file)

    """ Create a dictionary with playlist_id as key and name as value """
    id_name_dict = df.set_index('id').to_dict()['name']
    return id_name_dict

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


