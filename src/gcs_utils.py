import os
import glob
import json
import pickle
import logging
from google.cloud.storage import Bucket, Blob

from src.training.models.model_loader import ModelWrapper

logger = logging.getLogger(__name__)


def get_bucket(storage_client, config):
    # Set up GCS
    gcs_bucket = config["gcs_bucket"]["bucket"]
    bucket: Bucket = storage_client.bucket(gcs_bucket)
    return bucket

def dump_and_upload_file(gcs_client, config, gcs_path, file_path, file_name, data):
    parent_path = os.path.dirname(f"tmp_data/{file_name}")
    if len(glob.glob(parent_path)) == 0:
        os.makedirs(parent_path)

    pickle.dump(data, open(f"tmp_data/{file_name}", "wb"))

    bucket = get_bucket(gcs_client, config)

    file_blob = bucket.blob(f"{gcs_path}/{file_path}/{file_name}_")
    file_blob.upload_from_filename(f"tmp_data/{file_name}")
    logger.info(f"Uploaded file {file_name}")
    print(f"Uploaded file {file_name}")


def dump_and_upload_model(gcs_client, config, gcs_path, model: ModelWrapper):

    model_path = f"{model.info['name']}/{model.info['version']}/model.pkl"
    info_path = f"{model.info['name']}/{model.info['version']}/model_info.json"
    log_path = f"{model.info['name']}/{model.info['version']}/training.log"

    # Create parent directories if they don't exist
    parent_path = os.path.dirname(f"tmp_models/{model_path}")
    if len(glob.glob(parent_path)) == 0:
        os.makedirs(parent_path)
    print(os.getcwd())
    pickle.dump(model.model, open(f"tmp_models/{model_path}", "wb"))

    # Set up GCS
    bucket = get_bucket(gcs_client, config)

    # Upload model
    model_blob: Blob = bucket.blob(f"{gcs_path}/{model_path}")
    model_blob.upload_from_filename(f"tmp_models/{model_path}")
    logger.info("Uploading model")

    # Upload model_info.json
    info_blob = bucket.blob(f"{gcs_path}/{info_path}")
    logger.info(f"Upload model with info {model.info}")

    # dict to json file
    with open(f"tmp_models/{info_path}", "w") as model_info_file:
        json.dump(model.info, model_info_file)
    info_blob.upload_from_filename(f"tmp_models/{info_path}")
    logger.info("Model uploaded")

    # Upload training log
    log_blob = bucket.blob(f"{gcs_path}/{log_path}")
    log_blob.upload_from_filename("training.log")
