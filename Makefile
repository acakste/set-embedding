TRAINING_IMAGE= embedding-param-estimator
PARTITION_DASHED= 2024-01-01
CONFIGURATION_NAME= Param_comp_test_v1 #Deep_mean_v0 #min_dist_user_d256_1000t
VERSION= geomloss_quadratic
#mindist_user_data_d64


TEST_VAR=hello

PROJECT = syb-production-ai

GCS_BUCKET=syb-production-ai-test-clem

DOCKER_TRAINING_RUN_WITH_SA=docker run -ti \
                                      	-e GOOGLE_APPLICATION_CREDENTIALS=/app/secrets/$(PROJECT)/service-account.json \
                                      	$(TRAINING_IMAGE)

secrets/%/service-account.json:
	gcloud iam service-accounts keys create $@ \
          --iam-account anton-playground@$*.iam.gserviceaccount.com

build-docker-training:
	docker build -f Training.dockerfile -t $(TRAINING_IMAGE) .

build-docker-training-gpu:
	$(eval TRAINING_IMAGE=$(TRAINING_IMAGE)-gpu)
	docker build -f Training_gpu.dockerfile -t $(TRAINING_IMAGE) .

run_bash_entrypoint: secrets/$(PROJECT)/service-account.json build-docker-training
	$(DOCKER_TRAINING_RUN_WITH_SA) bash

run_test: secrets/$(PROJECT)/service-account.json build-docker-training
	$(DOCKER_TRAINING_RUN_WITH_SA) python src/training/pre_compute_dist.py

run_training_code: secrets/$(PROJECT)/service-account.json build-docker-training
	$(DOCKER_TRAINING_RUN_WITH_SA) python src/training/pipeline.py  --version "$(VERSION)-$(PARTITION_DASHED)" --configuration_name $(CONFIGURATION_NAME)

run_training_step: secrets/$(PROJECT)/service-account.json build-docker-training
	$(DOCKER_TRAINING_RUN_WITH_SA)  python src/training/data/dataloader.py

run_submit_training_code: secrets/$(PROJECT)/service-account.json build-docker-training-gpu
	$(DOCKER_TRAINING_RUN_WITH_SA) python src/create_job.py --project=$(PROJECT) --partition_date=$(PARTITION_DASHED) --location=us-central1 --staging_bucket=$(GCS_BUCKET) --display_name=vertex-ai-clem-training --container_uri=gcr.io/syb-production-ai/$(TRAINING_IMAGE)_gpu --service_account=anton-playground@syb-production-ai.iam.gserviceaccount.com --experiment=search-ranker-training --experiment_run=vertex-ai-clem-training-$(PARTITION_DASHED)-$(RUN_ID) --run_id=$(RUN_ID) --table_path=$(TABLE_PATH) --machine_type=n1-highmem-16 --accelerator_type=NVIDIA_TESLA_K80 --accelerator_count=2 --version=$(VERSION)-$(PARTITION_DASHED) --configuration_name $(CONFIGURATION_NAME)

tag_image:
	docker tag $(TRAINING_IMAGE):latest gcr.io/$(PROJECT)/$(TRAINING_IMAGE)

push_image: secrets/$(PROJECT)/service-account.json build-docker-training tag_image
	echo $(TRAINING_IMAGE)
	docker push gcr.io/$(PROJECT)/$(TRAINING_IMAGE)

test_gpu:
	$(eval TEST_VAR=$(TEST_VAR)-gpu)
test: test_gpu
	echo $(TEST_VAR)


