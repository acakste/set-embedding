from google.cloud import bigquery
import datetime
import yaml
import argparse

def schema_from_config_dict(config):
    """Create a bigquery schema from a dictionary using recursion"""
    schema = []
    for key, value in config.items():
        if isinstance(value, dict):
            nested_fields = schema_from_config_dict(value)
            schema.append(bigquery.SchemaField(key, field_type="RECORD", fields=nested_fields, mode="NULLABLE"))
        else:
            # Read python variable data type and convert to bigquery data type
            if isinstance(value, bool):
                field_type = "BOOLEAN"
            elif isinstance(value, int):
                field_type = "INTEGER"
            elif isinstance(value, float):
                field_type = "FLOAT"
            elif isinstance(value, str):
                field_type = "STRING"
            elif isinstance(value, datetime.date):
                field_type = "STRING"
            else:
                raise ValueError(f"Unsupported data type: {type(value)}")
            schema.append(bigquery.SchemaField(key, field_type=field_type, mode="NULLABLE"))

    return schema

def convert_yaml_to_schema(file_name):
    """Convert a yaml file to a bigquery schema. Create nested fields for nested dictionaries."""
    import yaml
    with open(file_name, 'r') as f:
        config = yaml.safe_load(f)

    schema = schema_from_config_dict(config)
    return schema

def insert_parameter_config_to_bigquery(configuration_name, yaml_files, project="syb-production-ai", dataset="anton_thesis", table="experiment_settings_V0"):
    """ Get the schema defined by the yaml files """
    inferred_schema = []
    inferred_schema.append(bigquery.SchemaField("configuration_name", field_type="STRING", mode="NULLABLE"))
    for file in yaml_files:
        file_schema = convert_yaml_to_schema(file)
        """ Get the file name from a string like 'folder/file_name.yaml' where it is unknown how many nested folders we have"""
        file_name = file.split("/")[-1].split(".")[0]  # The last element in the list is the file name, and we remove the file extension
        inferred_schema.append(bigquery.SchemaField(file_name, field_type="RECORD", fields=file_schema, mode="NULLABLE"))

    """ Get the existing schema from the table """
    bigquery_client = bigquery.Client(project=project)
    table = bigquery_client.get_table(f"{project}.{dataset}.{table}")

    """ Check if the schema is the same as the new schema """
    if table.schema != inferred_schema:
        """ Update the fields in the inferred schema that differ from the table schema"""

        table.schema = inferred_schema
        bigquery_client.update_table(table, ["schema"])

    """ Insert the configuration into the table """
    configuration_data = { "configuration_name": configuration_name}
    for file in yaml_files:
        with open(file, 'r') as f:
            config = yaml.safe_load(f)
        file_name = file.split("/")[-1].split(".")[0]  # The last element in the list is the file name, and we remove the file extension

        configuration_data[file_name] = config
    errors = bigquery_client.insert_rows_json(table, [configuration_data])
    print(f"errors: {errors}")

def check_existence_of_configuration(configuration_name, project="syb-production-ai", dataset="anton_thesis", table="experiment_settings_V0"):
    query = f"""
    SELECT *
    FROM `{project}.{dataset}.{table}`
    WHERE configuration_name = '{configuration_name}'"""

    bigquery_client = bigquery.Client(project=project)

    query_job = bigquery_client.query(query=query)
    result_df = query_job.to_dataframe()

    if len(result_df) > 0:
        return True
    else:
        return False

def read_experiment_config_from_bigquery(configuration_name, folder_for_yamls, project="syb-production-ai", dataset="anton_thesis", table="experiment_settings_V0"):
    query = f"""
    SELECT *
    FROM `{project}.{dataset}.{table}`
    WHERE configuration_name = '{configuration_name}'"""

    if check_existence_of_configuration(configuration_name) == False:
        raise ValueError(f"Configuration with the provided name {configuration_name} does not exist.")

    bigquery_client = bigquery.Client(project=project)
    query_job = bigquery_client.query(query=query)
    result_df = query_job.to_dataframe()

    print(result_df)
    param_dict = result_df.to_dict()

    # Loop through the dictionary and create yaml files from the names of the keys
    for key, value in param_dict.items():
        if key == "configuration_name":
            continue
        if len(value.items()) > 1:
            raise ValueError(f"Found more than 1 config file with the name {configuration_name}")

        for index, config in value.items():
            with open(f"{folder_for_yamls}/tmp_{key}.yaml", 'w') as f:
                yaml.dump(config, f)

def main():
    """
    Upload the current state of the parameters.yaml and experiment_config.yaml in src directory to BigQuery table.
    First check that no entry with the provided configuration_name exists.
    """
    parser = argparse.ArgumentParser(description='Generate a model from params and save it to cloud storage')
    parser.add_argument('--configuration_name', type=str, help='Name of the bigquery configuration target', required=True)
    args = parser.parse_args()

    # Check for an existing configuration with name configuration_name
    if check_existence_of_configuration(args.configuration_name) == True:
        raise ValueError(f"Configuration with the provided name {args.configuration_name} already exists.")

    # Upload the current parameters.yaml and experiment_config.yaml to BigQuery table.
    insert_parameter_config_to_bigquery(args.configuration_name, ["experiment_config.yaml", "parameters.yaml"])

if __name__ == "__main__":
    main()