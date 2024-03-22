

from src.training.data_handling.dataloader import DataLoader

import argparse, os
import asyncio

from src.training.data_handling.dataloader import DataLoader
from src.training.div.model_utils import set_config_files

async def download_user_playback(target_file_name, target_folder="../../../data/"):

    #----------- Comment if local
    parser = argparse.ArgumentParser(description='Generate a model from params and save it to cloud storage')

    parser.add_argument('--version', type=str, help='Model version. Default is "latest"', required=True)
    # parser.add_argument('--description', type=str, help='Model description', required=True)
    # parser.add_argument('--training_path', type=str, help="Path in cloud storage", required=True)
    parser.add_argument('--configuration_name', type=str, help='Name of the bigquery configuration target', required=True)
    parser.add_argument('--run_local', type=bool, help='Run locally or in docker', required=False, default=False)

    args = parser.parse_args()
    version = args.version or "latest"

    """ Download the configuration files from BigQuery and load them. """
    config, exp_config = set_config_files(args)

    if args.run_local == False:
        service_account_file = "/app/secrets/syb-production-ai/service-account.json"
    else:
        service_account_file = "../../../secrets/syb-production-ai/service-account.json"

    # Download the data set with labels
    data_loader = DataLoader(config=config["config"]["bigquery"], service_account_file=service_account_file)
    df = await data_loader.load_data(partition_date=exp_config['partition_date'])
    df.to_json(path_or_buf=target_folder + target_file_name)

def main():
    asyncio.run(download_user_playback())

if __name__ == "__main__":
    main()