
# ***********************
# Model Wrapper
import os
import json
import glob
import pickle
import logging
import xgboost as xgb
import typing
import torch.nn as nn
import torch

import google.cloud.storage as gcs
from google.cloud.storage import Blob, Bucket, Client as StorageClient


from src.training.models.models import SetTransformer, ParamEstimator
from src.training.models.deep_sets.modules import DeepSet
from src.training.models.OTKE.otk.models_deepsea import OTLayer
logger = logging.getLogger(__name__)


def load_from_pickle(model_file: str):
    logger.info(f"Loading model from folder {model_file}")
    model = pickle.load(open(model_file, "rb"))
    return model

""" function mapping for activations """
def activation_fn_map(activation_name):
    mapping = {"relu": nn.ReLU(),
               "sigmoid": nn.Sigmoid(),
               "tanh": nn.Tanh(),
               "leaky_relu": nn.LeakyReLU()}
    try:
        return mapping[activation_name]
    except KeyError:
        raise ValueError(f"Activation function {activation_name} not found in mapping.")

def pooling_op_fn_map(pooling_op_name):
    mapping = {"sum": torch.sum,
               "max": torch.max,
               "mean": torch.mean}
    try:
        return mapping[pooling_op_name]
    except KeyError:
        raise ValueError(f"Pooling operation {pooling_op_name} not found in mapping.")

class ModelWrapper:
    def __init__(self, info: dict, model=None):
        self.info = info
        self.model = model
    def create_model(self, model_type: str, exp_config: dict):
        model_parameters = exp_config[model_type]

        if model_type == "XGBoost":
            self.model = xgb.XGBClassifier(scale_pos_weight=model_parameters["scale_pos_weight"])
        elif model_type == "SetTransformer":
            self.model = SetTransformer(input_dim=model_parameters["input_dim"],
                                        output_dim=model_parameters["output_dim"],
                                        hidden_dim=model_parameters["hidden_dim"],
                                        num_heads=model_parameters["num_heads"],
                                        num_inds=model_parameters["num_inds"],
                                        num_seeds=model_parameters["num_seeds"],
                                        num_enc_layers=model_parameters["num_enc_layers"],
                                        num_dec_layers=model_parameters["num_dec_layers"],
                                        use_ISAB=model_parameters["use_ISAB"])
        elif model_type == "ParamEstimator":
            self.model = ParamEstimator(input_dim=model_parameters["input_dim"],
                                        hidden_dim=model_parameters["hidden_dim"],
                                        embedding_dim=model_parameters["embedding_dim"],
                                        num_components=model_parameters["num_components"]
            )
        elif model_type == "DeepSet":

            self.model = DeepSet(input_dim=model_parameters["input_dim"],
                                 output_dim=model_parameters["output_dim"],
                                 hidden_dim=model_parameters["hidden_dim"],
                                 pooling_op=pooling_op_fn_map(model_parameters["pooling_op"]),
                                 num_enc_layers=model_parameters["num_enc_layers"],
                                 num_dec_layers=model_parameters["num_dec_layers"],
                                 hidden_activation=activation_fn_map(model_parameters["hidden_activation"]),
                                 output_activation=activation_fn_map(model_parameters["output_activation"])
            )
        elif model_type == "OTKE":
            self.model = OTLayer(in_dim=100, out_size=100, heads=1, eps=0.1, max_iter=10)
            raise NotImplementedError
        else:
            raise ValueError(f"model_type: {model_type} is not a valid model.")

# *********************

def get_model_name_from_path(model_folder: str):
    path_split = model_folder.split("/")
    # expected: ['models', 'name', 'version']
    assert path_split[0] == "models"
    name = path_split[1]
    version = path_split[2]
    return name, version

class ModelLoader:
    def __init__(self, gcs_client, bucket: Bucket):
        self.gcs_client = gcs_client
        self.bucket = bucket
        self.model_loaders = {"pickle": load_from_pickle}

    def list_available_models(self, prefix: str):
        blobs_in_models = self.gcs_client.list_blobs(bucket_or_name=self.bucket, prefix=prefix)
        all_blobs_in_models: typing.List[Blob] = list(blobs_in_models)
        # Append found model_folders here
        paths = []
        for blob_in_models in all_blobs_in_models:
            paths.append(blob_in_models.name)

        names_with_versions = {}
        for path in paths:
            name, version = get_model_name_from_path(path)
            if name in names_with_versions and version not in names_with_versions[name]:
                names_with_versions[name].append(version)
            else:
                names_with_versions[name] = [version]
        return names_with_versions

    def download_model(self, gcs_path, model_name, model_version, download_folder):

        # Download model
        model_folder = f"{gcs_path}/{model_name}/{model_version}"
        model_specific_blobs = self.gcs_client.list_blobs(bucket_or_name=self.bucket, prefix=model_folder)
        download_folder = download_folder or "./"

        logger.info(f"Downloading from folder {model_folder}")

        for blob in model_specific_blobs:
            download_path = os.path.join(download_folder, blob.name)

            # Create parent directories if they don't exist
            parent_path = os.path.dirname(download_path)
            if len(glob.glob(parent_path)) == 0:
                os.makedirs(parent_path)

            logger.info(f"Downloading model {model_name} to path {download_path}")
            blob.download_to_filename(filename=download_path)
            logger.info(f"Downloaded model {model_name} to path {download_path}")

        downloaded_model_folder = f"{download_folder}{model_folder}"
        return downloaded_model_folder

    def load_model(self, model_path, model_name, model_version) -> ModelWrapper:

        loader = self.model_loaders["pickle"]
        full_model_path = model_path + f"{model_name}/{model_version}/"
        # training/downloads/models/XGBoost/latest/model.pkl
        try:
            trained_model = loader(model_file=full_model_path + "model.pkl")
            with open(full_model_path + "/model_info.json", "r") as model_info_json:
                info = json.load(model_info_json)
            model = ModelWrapper(info=info, model=trained_model)
            print(f"Model {model.info['name']} with version {model.info['version']} loaded")
        except Exception as e:
            logger.error("Could not load model under path {}. {}".format(model_path, e))
            model = None

        return model

