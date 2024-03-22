
import pandas as pd
import torch
from google.cloud import storage
import datetime
from geomloss import SamplesLoss

from time import time

from src.training.models.model_loader import ModelWrapper, ModelLoader
from src.training.models.training import ModelTrainer
from src.training.data_handling.dataloader import DataLoader
from src.training.data_handling.preprocesser import Preprocesser
from src.training.distortion.dist_metrics import wasserstein

from src.gcs_utils import dump_and_upload_model, get_bucket, dump_and_upload_file



def pre_compute_dist(dataset, logger=None):
    """ Precompute all pairwise distances for the dataset. """
    pass

    N = len(dataset)
    dist_matrix = torch.zeros(size=(N,N))

    if logger != None:
        if torch.cuda.is_available():
            logger.info("Using GPU for computing distances.")
        else:
            logger.info("No GPU found. Using CPU for computing distances.")

    # Corresponds to the Wasserstein 2-distance.
    dist_func = SamplesLoss(loss="sinkhorn", p=2, blur=0.05)

    for i in range(N):
        for j in range(i + 1, N):
            x_1 = dataset[i][0].contiguous()
            x_2 = dataset[j][0].contiguous()

            logger.info(f"Computing distance between instance {i} and instance {j}, (L_1,L_2) = ({x_1.shape[0]},{x_1.shape[0]})")
            if torch.cuda.is_available():
                x_1 = x_1.cuda()
                x_2 = x_2.cuda()

            # dist = wasserstein(x_1, x_2, verbose=False, max_iters=3, p=4)
            dist = dist_func(x_1, x_2)
            dist_matrix[i,j] = dist
            dist_matrix[j,i] = dist

    label_id_mapping = {"id2label":{}, "label2id":{}}
    for id, (_, label) in enumerate(dataset):
        label_id_mapping["id2label"][id] = label
        label_id_mapping["label2id"][label] = id

    return dist_matrix, label_id_mapping

def get_label_id_mapping(dataset):
    label_id_mapping = {"id2label":{}, "label2id":{}}
    for id, (_, label) in enumerate(dataset):
        label_id_mapping["id2label"][id] = label
        label_id_mapping["label2id"][label] = id
    return label_id_mapping

class Experiment:
    def __init__(self, config: dict, exp_config: dict, version: str, configuration_name, logger, service_account_file: str = None):
        self.config = config
        self.exp_config = exp_config
        self.configuration_name = configuration_name
        self.version = version
        self.gcs_client = storage.Client(project=config["gcs_bucket"]["project"])
        self.logger = logger
        self.service_account_file = service_account_file

    async def run_pre_compute_dist(self):
        self.logger.info(f"Starting run {self.exp_config['name']} for partition {self.exp_config['partition_date']}")
        data_loader = DataLoader(config=self.config["config"]["bigquery"], service_account_file=self.service_account_file)

        df = await data_loader.load_data(partition_date=self.exp_config['partition_date'])

        self.logger.info("Preprocessing data")
        preprocesser = Preprocesser(input_df=df, groupby_str=self.config["config"]["bigquery"]["groupby_str"])
        dataset = preprocesser.preprocess()

        self.logger.info(f"Preprocessed data has {len(dataset)} instances.")

        dist_matrix, label_id_mapping = pre_compute_dist(dataset, logger=self.logger)
        self.dist_matrix = dist_matrix
        self.label_id_mapping = label_id_mapping

    async def run_experiment(self, use_precomputed_dist=False):
        self.logger.info(f"Starting run {self.exp_config['name']} for partition {self.exp_config['partition_date']}")
        data_loader = DataLoader(config=self.config["config"]["bigquery"], service_account_file=self.service_account_file)
        df = await data_loader.load_data(partition_date=self.exp_config['partition_date'])

        self.logger.info("Preprocessing data")
        preprocesser = Preprocesser(input_df=df, groupby_str=self.config["config"]["bigquery"]["groupby_str"])
        dataset = preprocesser.preprocess()
        self.logger.info(f"Preprocessed data has {len(dataset)} instances.")

        if use_precomputed_dist:
            """ Better to do this in a separate pipeline. And just load an existing file in that case."""
            raise NotImplementedError

        """ Create the model based on exp_config."""
        self.logger.info(f"Creating model {self.exp_config['model_type']}")
        model = ModelWrapper(info={"name": f"{self.exp_config['model_type']}",
                                   "version": self.version,
                                   "date": datetime.datetime.now().strftime("%Y-%m-%d"),
                                   "model_type": self.exp_config['model_type'],
                                   "model_parameters": self.exp_config[self.exp_config['model_type']]})
        model.create_model(model_type=self.exp_config['model_type'], exp_config=self.exp_config)

        """ Train the model. """
        self.logger.info(f"Training model {self.exp_config['model_type']}")
        tick = time()
        model_trainer = ModelTrainer(model=model, data=dataset, exp_config=self.exp_config, precomputed_dist=None, label_id_mapping=None)
        model_trainer.train()
        training_time = time() - tick
        self.logger.info(f"Training took {training_time} s.")

        """ Dump and upload the trained model. This function also uploads the log file."""
        dump_and_upload_model(gcs_client=self.gcs_client,
                              config=self.config,
                              gcs_path=self.exp_config['gcs_path'],
                              model=model)
