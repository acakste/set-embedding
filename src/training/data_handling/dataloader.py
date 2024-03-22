import yaml
import logging
import asyncio
from dataclasses import dataclass
from google.cloud import bigquery
from google.oauth2 import service_account



from src.training.data_handling.data_collection import get_embeddings
logger = logging.getLogger(__name__)


@dataclass
class DataLoader:
    def __init__(self, config: dict, service_account_file: str):
        self.config = config
        if service_account_file != None:
            credentials = service_account.Credentials.from_service_account_file(service_account_file)
            self.client = bigquery.Client(credentials=credentials)
            logger.info("Using service account for DataLoader client.")
        else:
            self.client = bigquery.Client(project="syb-production-ai")

    def get_query(self, partition_date: str):
        return f"""
        SELECT
         *
        FROM `{self.config['project']}.{self.config['training_dataset']}.{self.config['training_table']}`
        """

    async def load_data(self, partition_date=None):
        return self.load_bigquery_data(partition_date=partition_date)

    def load_bigquery_data(self, partition_date: str):
        logger.info(f"Loading data from BigQuery for partition {partition_date}")
        query = self.get_query(partition_date)
        query_job = self.client.query(query)
        return query_job.to_dataframe()

    def load_file_data(self, file_path: str, file_name:str):
        logger.info(f"Loading data from file: {file_name}")
        print("WARNING: ugly, temporary get_embeddings loading")

        return get_embeddings(embedding_query="top_readymades_embedding")



if __name__ == '__main__':

    import os
    print(os.getcwd())
    #os.chdir("")

    conf_path = "/app/src/parameters.yaml"
    with open(conf_path, 'r') as f:
        config = yaml.safe_load(f)
    data_loader = DataLoader(config=config["config"]["bigquery"])
    df = asyncio.run(data_loader.load_data(partition_date="2021-01-01"))
    print(df.head())
