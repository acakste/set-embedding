import os
import yaml
import pandas as pd
from google.cloud import storage

from src.bigquery_param_setting import read_experiment_config_from_bigquery
from src.training.models.model_loader import ModelLoader
from src.gcs_utils import dump_and_upload_model, get_bucket, dump_and_upload_file
from src.training.data_handling.data_utils import extract_sets


def set_config_files(args, logger=None):
    """Fetch and create the configuration files according to args.configuration_name.
    Returns the configuration and experiment configuration as dictionaries."""
    if args.run_local == False:
        # make sure the folder exists, if not create it
        if not os.path.exists("/app/src/training/_config_files"):
            os.makedirs("/app/src/training/_config_files")
        folder_for_yamls = "/app/src/training/_config_files"
    else:
        if not os.path.exists("config_files"):
            os.makedirs("config_files")
        folder_for_yamls = "config_files"

    read_experiment_config_from_bigquery(args.configuration_name, folder_for_yamls=folder_for_yamls)

    if logger != None:
        logger.info(f"Configuration file '{args.configuration_name}' loaded.")
        logger.info("Logging configuration")

    if args.run_local == False:
        conf_path = "/app/src/training/_config_files/tmp_parameters.yaml"
    else:
        conf_path = "config_files/tmp_parameters.yaml"

    with open(conf_path, 'r') as f:
        config = yaml.safe_load(f)

    if logger != None:
        logger.info("Logging experiment configuration")
    if args.run_local == False:
        """ Dockerized """
        exp_conf_path = "/app/src/training/_config_files/tmp_experiment_config.yaml"
    else:
        """ Local """
        exp_conf_path = "config_files/tmp_experiment_config.yaml"

    with open(exp_conf_path, 'r') as f:
        exp_config = yaml.safe_load(f)

    return config, exp_config


def load_model_from_gcs(args, model_version, model_name, also_download=False):
    config, exp_config = set_config_files(args)

    gcs_client = storage.Client(project=config["gcs_bucket"]["project"])

    model_loader = ModelLoader(gcs_client=gcs_client,
                               bucket=get_bucket(gcs_client, config))

    available_models = model_loader.list_available_models(prefix="models/")
    print(f"Available models are {available_models}")

    if also_download == True:
        # Download models from GCS
        download_folder = model_loader.download_model(gcs_path=exp_config['gcs_path'],
                                                      model_name=model_name,
                                                      model_version=model_version,
                                                      download_folder="../../trained_models/")

    # Load local model
    local_model = model_loader.load_model(model_path="../../trained_models/models/",
                                          model_name=model_name,
                                          model_version=model_version)
    return local_model.model


def embed_and_download(args, model_name, model_version, run_local=True):
    raise NotImplementedError
    # args = parser.parse_args()
    version = "latest"

    # print(args)
    # print(args.run_local)
    """Fetch and create the configuration files"""
    if run_local == False:
        # make sure the folder exists, if not create it
        if not os.path.exists("/app/src/training/config_files"):
            os.makedirs("/app/src/training/config_files")
        folder_for_yamls = "/app/src/training/config_files"
    else:
        if not os.path.exists("config_files"):
            os.makedirs("config_files")
        folder_for_yamls = "config_files"

    read_experiment_config_from_bigquery(args.configuration_name, folder_for_yamls=folder_for_yamls)

    """ Load the model. """
    model = load_model_from_gcs(args, model_version, model_name)

    """ Load the data. """
    df = pd.read_json(path_or_buf="../../data/user_playback_balanced_v3.json")
    dataset = extract_sets(df, groupby_str="account_id")

    """Embed the data. Save embeddings. """
    embeddings = []
    labels = []
    for i in range(len(dataset)):
        x, label = dataset[i]
        embeddings.append(model.embed(x.unsqueeze(0)))
        labels.append(label)

