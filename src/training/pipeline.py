import pandas as pd
import yaml
import logging
import argparse
import asyncio
import os
from time import time

from src.training.experiment import Experiment
from src.training.div.model_utils import set_config_files
from src.bigquery_param_setting import read_experiment_config_from_bigquery

logging.basicConfig(
    level=logging.INFO,
    filename="training.log",
    filemode="w",
    format="%(asctime)-15s %(message)s")

logger = logging.getLogger(__name__)

pandas_logger = logging.getLogger('pd')
pandas_logger.setLevel(logging.ERROR)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a model from params and save it to cloud storage')

    # parser.add_argument('--description', type=str, help='Model description', required=True)
    # parser.add_argument('--training_path', type=str, help="Path in cloud storage", required=True)

    parser.add_argument('--version', type=str, help='Model version. Default is "latest"', required=True)
    parser.add_argument('--configuration_name', type=str, help='Name of the bigquery configuration target', required=True)
    parser.add_argument('--run_local', action='store_true', help='Run locally or in docker')
    parser.set_defaults(run_local=False)

    args = parser.parse_args()
    version = args.version or "latest"

    config, exp_config = set_config_files(args, logger=logger)

    if args.run_local == False:
        service_account_file = "/app/secrets/syb-production-ai/service-account.json"
    else:
        service_account_file = "../../secrets/syb-production-ai/service-account.json"

    experiment = Experiment(config, exp_config, version=version, configuration_name=args.configuration_name, logger=logger, service_account_file=service_account_file)
    asyncio.run(experiment.run_experiment())

    """ Probably create a completely separate pipeline for this."""
    # asyncio.run(experiment.run_pre_compute_dist())

