from typing import Optional
from google.cloud import aiplatform
import argparse
import time
import os


def create_custom_job(
        project: str,
        location: str,
        staging_bucket: str,
        display_name: str,
        container_uri: str,
        run_id: str,
        table_path: str,
        partition_date: str,
        machine_type: str,
        accelerator_type: str,
        accelerator_count: str,
        version: str,
        configuration_name: str
) -> None:
    aiplatform.init(project=project, location=location, staging_bucket=staging_bucket)
    machine_spec = {"machine_type": machine_type}

    if accelerator_type and accelerator_type.lower() != "none":
        machine_spec["accelerator_type"] = accelerator_type
        if accelerator_count.isnumeric():
            machine_spec["accelerator_count"] = int(accelerator_count)

    job: aiplatform.CustomJob = aiplatform.CustomJob(
        display_name=display_name,
        worker_pool_specs=[
            {
                "machine_spec": machine_spec,
                "replica_count": 1,
                "container_spec": {
                    "image_uri": container_uri,
                    "command": ["python", "src/training/pipeline.py"],
                    "args": ["--version", version,
                            "--configuration_name", configuration_name],
                    # ["--partition_date", partition_date,
                          #   "--version", version],
                },
            }
        ]
    )
    job.run()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project',
                        help='Project to use',
                        required=True)
    parser.add_argument('--location',
                        help='Location to use',
                        required=True)
    parser.add_argument('--staging_bucket',
                        help='Staging bucket to use',
                        required=True)
    parser.add_argument('--display_name',
                        help='Display name to use',
                        required=True)
    parser.add_argument('--container_uri',
                        help='Container URI to use',
                        required=True)
    parser.add_argument('--run_id',
                        help='Run id to use',
                        required=True)
    parser.add_argument('--table_path',
                        help='Table path to use',
                        required=True)
    parser.add_argument('--partition_date',
                        help='Partition date to use',
                        required=True)
    parser.add_argument('--machine_type',
                        help='Machine type to use',
                        required=True)
    parser.add_argument('--accelerator_type',
                        help='Accelerator type to use',
                        required=True)
    parser.add_argument('--accelerator_count',
                        help='Accelerator count to use',
                        required=True)
    parser.add_argument('--version',
                        help='Version to save model as',
                        required=True)
    parser.add_argument('--configuration_name',
                        help='Which configuration to fetch from experiment_settings BigTable',
                        required=True)
    known_args, pipeline_args = parser.parse_known_args()

    create_custom_job(
        project=known_args.project,
        location=known_args.location,
        staging_bucket=known_args.staging_bucket,
        display_name=known_args.display_name,
        container_uri=known_args.container_uri,
        run_id=known_args.run_id,
        table_path=known_args.table_path,
        partition_date=known_args.partition_date,
        machine_type=known_args.machine_type,
        accelerator_type=known_args.accelerator_type,
        accelerator_count=known_args.accelerator_count,
        version=known_args.version,
        configuration_name=known_args.configuration_name
    )

if __name__ == '__main__':
    main()