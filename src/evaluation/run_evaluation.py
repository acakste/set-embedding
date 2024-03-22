import numpy as np

from held_out_attributes import distance_based_attribute_evaluation, classification_based_attribute_evalutation, multilabel_classification_based_attribute_evaluation, distance_based_attribute_evaluation
from src.training.models.model_loader import ModelWrapper
from src.evaluation.held_out_attributes import get_attribute_labels
from src.other.set_genre_tag_setting import get_genre_tag_from_track_set
from src.training.div.utils import count_parameters

import torch
from src.training.models.models import SetTransformer, ParamEstimator
from src.other.create_dataset import create_attribute_dataset
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt



# New imports
import warnings

from src.training.div.model_utils import set_config_files, load_model_from_gcs
from src.training.data_handling.data_utils import extract_sets
from src.training.distortion.min_distortion import select_distortion_function

from src.evaluation.distortion_distribution import train_test_distortion, estimate_distortion_distribution
from src.evaluation.clustering import cluster_embedding, reduce_user_embeddings




class ModelEvaluation:
    def __init__(self, model, trainset, testset, eval_config):
        self.model = model
        self.eval_config = eval_config

        # Unpack the train and test dataset into input and labels
        (x_train, labels_train) = zip(*trainset)
        self.x_train = x_train
        self.labels_train = labels_train

        (x_test, labels_test) = zip(*testset)
        self.x_test = x_test
        self.labels_test = labels_test

        # Fetch the function pointer for the distortion function
        self.distortion_func = select_distortion_function(eval_config["distortion"]["distortion_func"])


    def run(self):
        """ Might be possible to do. But could be a little tricky since we might want to do different files and such."""
        raise NotImplementedError

    def embed(self):
        """ Embed the datasets """
        train_embeddings = []
        for x in self.x_train:
            train_embeddings.append(self.model(x.unsqueeze(0)).squeeze(0).detach())
        self.X_embedding_train = torch.stack(train_embeddings)

        test_embeddings = []
        for x in self.x_test:
            test_embeddings.append(self.model(x.unsqueeze(0)).squeeze(0).detach())
        self.X_embedding_test = torch.stack(test_embeddings)




    def distortion_based_evaluation(self):
        # Get the average train/test distortion and make figure of a sampled train and test distortion
        avg_train_distortion, avg_test_distortion = estimate_distortion_distribution(self.x_train,
                                                                                     self.x_test,
                                                                                     self.X_embedding_train,
                                                                                     self.X_embedding_test,
                                                                                     self.distortion_func,
                                                                                     self.eval_config["distortion"]["sample_size"],
                                                                                     self.eval_config["distortion"]["max_num_tracks"])

        return avg_train_distortion, avg_test_distortion,

    def distribution_over_tags_evaluation(self):
        raise NotImplementedError
        pass

    def attribute_evaluation(self, X, X_labels, attribute_file, column_name, groupby_str):
        """
        :param X: The embedding to be used, can be either from the train or test set
        :param attribute_file: The file path containing the attribute labels
        :param column_name: Which column to use for finding and setting the attribute labels
        :param groupby_str: The groupby_str.
        :return:
        """

        # Set pandas printing options
        pd.set_option('display.max_columns', 15, 'display.width', 2500, 'display.max_rows', 100, 'display.max_colwidth', 1000)

        # Load the attribute file
        df_attribute = pd.read_json(attribute_file)
        label_ids, attribute_names, attribute_labels = get_attribute_labels(df_attribute, column_name=column_name, groupby_str=groupby_str, are_grouped=False)

        n_labeled_embeddings = len(set(X_labels).intersection(label_ids))
        if n_labeled_embeddings < len(X_labels):
            warnings.warn("The number of labeled instances is less than the number of instances in the dataset.")

        if n_labeled_embeddings != len(label_ids):
            raise ValueError("The number of instances.")


        _, d_embed = X.shape
        labeled_embeddings = torch.zeros(size=(n_labeled_embeddings, d_embed))
        for i in range(n_labeled_embeddings):
            # find the index of the label in the label_ids
            index = X_labels.index(label_ids[i])
            labeled_embeddings[i] = X[index]


        #--------------------- Multi-class ---------------------
        do_multiclass = True
        y_label = []
        # Check that every attribute_labels is a one-hot vector
        for i in range(len(attribute_labels)):
            _tmp = torch.argmax(attribute_labels[i], dim=0)
            if not (attribute_labels[i][_tmp] == 1. and torch.sum(attribute_labels[i]) == 1.):
                do_multiclass = False
                break
            y_label.append(int(_tmp))
        y_label = torch.tensor(y_label)

        if do_multiclass == True:
            """ Do multi-class classification for one-hot vectors. """
            metrics, error_code = classification_based_attribute_evalutation(X=labeled_embeddings.detach(), y_label=y_label, n_splits=self.eval_config["attribute"]["n_splits"], is_binary=False)
            if error_code != 0:
                raise ValueError
            confusion_matrix = metrics["confusion_matrix"]
            mean_acc = metrics["accuracy_mean"]
            std_acc = metrics["accuracy_std"]
            print(f"Classification Accuracy: {mean_acc}, \n Standard Deviation: {std_acc}")
            print(f"Confusion matrix: \n{confusion_matrix}")
        else:
            print("TODO: multi-class logistic regression.")
            raise NotImplementedError


        #--------------------- Binary-class ---------------------

        """ Do Binary-class classification. One vs All for each attribute."""
        df_results = multilabel_classification_based_attribute_evaluation(X=labeled_embeddings.detach(), y_label=attribute_labels.detach(), attribute_names=attribute_names, n_splits=self.eval_config["attribute"]["n_splits"])

        disp = ConfusionMatrixDisplay(confusion_matrix, display_labels=None).plot(cmap="Blues")
        print(df_results)


        #--------------------- Attribute Average Distance ---------------------

        # Inter/intra attribute distance
        print(distance_based_attribute_evaluation(X=labeled_embeddings.detach(), y=attribute_labels.detach(), attribute_names=attribute_names))
        print("\n\n")

        print("TODO: See how distortion is distributed instead of just the average.")


        return

    def cluster_evaluation(self, X):
        # Cluster the embedding using HDSCAN
        clusterer = cluster_embedding(X)

        # Reduce the embeddings and plot with the clusters, just for visualization
        reduce_user_embeddings(X,
                               clusterer.labels_,
                               n_neighbors=self.eval_config["clustering"]["n_neighbors"],
                               min_dist=self.eval_config["clustering"]["n_neighbors"])

        # See how attributes are distributed over the clusters
        print("TODO: See how attributes are distributed over the clusters.")
        #   We want to see:
        #       * Do the instances of one cluster all share one (or multiple) attribute(s)?
        #       * How are the attributes distributed over the clusters?


        # See the ratio of Silhoutte score on the train/test set
        # Or maybe, Davies-Boudin index?

        # NOTE : I guess the Silhoutte score in itself is not interesting. If the embedding model just pushes instances
        # to one of the corners of a hypercube, then we will get a high Silhoutte score, but it is not a meaningful embedding.




async def evaluation_pipe():
    """Parse arguments"""
    parser = argparse.ArgumentParser(description='Generate a model from params and save it to cloud storage')

    parser.add_argument('--version', type=str, help='Model version. Default is "latest"', required=True)
    parser.add_argument('--configuration_name', type=str, help='Name of the bigquery configuration target', required=True)
    parser.add_argument('--run_local', action='store_true', help='Run locally or in docker')
    parser.add_argument('--local_data', action='store_true', help='Use local data or query table')
    parser.set_defaults(run_local=False, local_data=False)

    args = parser.parse_args()
    version = args.version or "latest"

    """Fetch and create the configuration files"""
    config, exp_config = set_config_files(args, logger=None)

    """ Might be needed for some of the evaluation. """
    if args.run_local == False:
        service_account_file = "/app/secrets/syb-production-ai/service_account.json"
    else:
        service_account_file = "../../secrets/syb-production-ai/service_account.json"

    # Load local model
    model_version = "geomloss_quadratic-2024-01-01"
    model_name = "SetTransformer"
    model = load_model_from_gcs(args, model_version, model_name)


    """ Load a eval_config file """
    with open("eval_config.yaml") as f:
        eval_config = yaml.safe_load(f)

    # Load the data set with labels
    if args.local_data == True:
        # df = pd.read_json(path_or_buf="../../data/top_readymades_embedding.json")
        df_train = pd.read_json(path_or_buf="../../data/user_playback_balanced_v3.json")
        df_test = pd.read_json(path_or_buf="../../data/user_playback_balanced_v3.json")
    else:
        raise NotImplementedError

    trainset = extract_sets(df_train, groupby_str="account_id")
    testset = extract_sets(df_test, groupby_str="account_id")


    """ Create the model evaluation object """

    model_evaluation = ModelEvaluation(model, trainset, testset, eval_config)
    model_evaluation.embed()

    """ Run the different types of evaluation """

    # 1. Distortion based evaluation
        # How do we do with test set and train set? Maybe they are separate from the start? But should the other evaluations
        # be done only on the test set?

    avg_train_distortion, avg_test_distortion = model_evaluation.distortion_based_evaluation()
    print(f"Average test distortion: {avg_test_distortion}, \n Average train distortion: {avg_train_distortion}")

    # 2. Attribute based evaluation:
    #       *   Attributes are hard labels, but a single instances can have multiple attributes.
    #           If instances have multiple attributes, no attempt to do multi-class classification is made.
    #       *   The input (train_set or test_set) is split into test and train set, to fit and predict the attributes.

    trainset_attribute_file = "../../data/user_playback_balanced_v3.json"
    testset_attribute_file = None
    # column_name = "genre_tags_list"

    # Do the evaluation on the train set, that is the instances that the embedding model has seen.
    model_evaluation.attribute_evaluation(model_evaluation.X_embedding_train,
                                          model_evaluation.labels_train,
                                          trainset_attribute_file,
                                          eval_config["attribute"]["column_name"],
                                          groupby_str=eval_config["attribute"]["groupby_str"])



    # Do the evaluation on the test set, that is the instances that the embedding model has not seen.
    if testset_attribute_file != None:
        pass
        # model_evaluation.attribute_evaluation(model_evaluation.X_embedding_test, model_evaluation.labels_test, testset_attribute_file, "business_type", groupby_str=eval_config["groupby_str"])


    # 3. Distribution over tags evaluation
    #       * Try to predict the distribution over some tags/attributes for each instance.
    # model_evaluation.distribution_over_tags_evaluation()


    """ clustering """
    # Can we fit the cluster model on the train set, and then see the ratio of Silhoutte score on the train/test set?
    # This and also see how our attributes are distributed over the clusters?
    # model_evaluation.cluster_evaluation(model_evaluation.X_embedding_train)


    """ recall@k for playlists"""
    # The problem with recall@k for playlists is that we do it by similarity search, we might get a lot of similar playlists
    # in the top results, although the user has only played one of them. This is unfair evaluation.
    # Better is maybe to predict if the user has played this playlist or not. Based on the user embedding and playlist embedding?


    """ similarity search (users, and playlists?)"""










def test_evaluation_gaussian():
    model = SetTransformer(input_dim=3, output_dim=3, num_dec_layers=1)
    model.load_state_dict(torch.load("../../trained_models/test_2"))

    # model = ParamEstimator(input_dim=3, hidden_dim=64, embedding_dim=3, num_components=2)
    # model.load_state_dict(torch.load("../../trained_models/test_paramestimator"))
    model.eval()

    """ Real gaussian data"""
    dataset = create_attribute_dataset(B=1000, L=200, d=3, num_gaussians=4, num_components=1)
    x, y_labels = zip(*dataset)

    x = torch.stack(x)
    y_labels = torch.stack(y_labels)
    print(y_labels.shape)
    print(x.shape)

    x_embed = model(x)

    attribute_names = [str(i) for i in range(len(y_labels[0]))]
    df = distance_based_attribute_evaluation(x_embed.detach(), y_labels.detach(), attribute_names)

    pd.options.display.max_columns = None
    pd.options.display.width = 2000

    print(df)

def test_evaluation_classifier_gaussian():
    model = SetTransformer(input_dim=3, output_dim=3, num_dec_layers=1)
    model.load_state_dict(torch.load("../../trained_models/test_2"))

    # model = ParamEstimator(input_dim=3, hidden_dim=64, embedding_dim=3, num_components=2)
    # model.load_state_dict(torch.load("../../trained_models/test_paramestimator"))
    model.eval()

    """ Real gaussian data"""
    num_gaussians = 30
    dataset = create_attribute_dataset(B=1000, L=100, d=3, num_gaussians=num_gaussians, num_components=2, uniform_components=False, threshold=0.2)
    x, y_labels = zip(*dataset)
    x = torch.stack(x)
    y_labels = torch.stack(y_labels)
    # y_labels = (torch.randn(size=(1000, num_gaussians)) > 0.3)*1.
    y = torch.argmax(y_labels, dim=1)
    print(y.shape)
    print(x.shape)

    """ Random data """
    # x = torch.randn(size=(1000, 200, 3))
    # y = torch.randint(0, 100, size=(1000,))
    # print(y.shape)

    x_embed = model(x)



    # mean_acc, std_acc = classification_based_attribute_evalutation(x_embed.detach(), y.detach())
    # print(mean_acc, std_acc)


    """For multi-label classification, one at a time"""
    attribute_names = [str(i) for i in range(num_gaussians)]
    df = multilabel_classification_based_attribute_evaluation(x_embed.detach(), y_labels.detach(), attribute_names, n_splits=5)
    pd.options.display.max_columns = None
    pd.options.display.width = 2000
    print(df)



import argparse, os, logging
from src.bigquery_param_setting import read_experiment_config_from_bigquery
import yaml
import asyncio

from src.training.models.model_loader import ModelLoader
from src.gcs_utils import dump_and_upload_model, get_bucket, dump_and_upload_file
from google.cloud import storage

from src.training.data_handling.dataloader import DataLoader
from src.training.data_handling.preprocesser import Preprocesser

from src.evaluation.held_out_attributes import classification_based_attribute_evalutation



def test_business_type_evaluation(df, local_model, dataset, exp_config):
    (x, labels) = zip(*dataset)

    """ Business type based attributes 
        _______________________________
    """
    # Get the columns values from df under the business_type column
    attribute_names = list(df['business_type'].unique())

    attributes = []
    for label in labels:
        # Get business type for account_id equal to label
        business_type_series = df.loc[df['account_id'] == label, 'business_type']

        # Check that every column value for each label is the same, if not raise an error
        if business_type_series.nunique() != 1:
            raise ValueError("The business type for the account_id is not the same for all the rows")

        # Get the unique business type
        business_type = business_type_series.unique()

        # Get the index from attribute_names
        # get the index from the list attribute_names
        index = attribute_names.index(business_type)
        attributes.append(index)

    # Convert attributes to a tensor with one hot vectors
    # attributes = torch.nn.functional.one_hot(torch.tensor(attributes))
    attributes = torch.tensor(attributes)
    """ End of business type based attributes
    _______________________________
    """


    # Use the model
    model = local_model.model
    model.eval()

    embeddings = torch.zeros(size=(len(dataset), exp_config[exp_config["model_type"]]["output_dim"]))

    # Get the embeddings
    lengths = torch.zeros(size=(len(dataset),))
    for i in range(len(dataset)):
        x_input, label = dataset[i]
        embeddings[i] = model(x_input.unsqueeze(0))
        lengths[i] = x_input.shape[0]



    """ Save the embeddings """

    print(labels)
    print(embeddings.shape)
    print([attribute_names[attributes[i]] for i in range(len(attributes))])
    print(attributes)
    df_result = pd.DataFrame({"account_id": labels, "business_type": [attribute_names[attributes[i]] for i in range(len(attributes))], "embedding": list(embeddings.detach().numpy())})
    df_result.to_json(path_or_buf="../../data/embeddingsV0_d128_4types.json")

    """ Do the evaluation (classification & distance based) """
    pd.set_option('display.max_columns', 15)
    pd.set_option('display.width', 2500)

    y_label = attributes

    metrics, error_code = classification_based_attribute_evalutation(X=embeddings.detach(), y_label=y_label.detach(), n_splits=5, id_labels=labels, is_binary=False)
    if error_code != 0:
        raise ValueError
    confusion_matrix = metrics["confusion_matrix"]
    mean_acc = metrics["accuracy_mean"]
    std_acc = metrics["accuracy_std"]

    df_results = multilabel_classification_based_attribute_evaluation(X=embeddings.detach(), y_label=torch.nn.functional.one_hot(y_label).detach(), attribute_names=attribute_names, n_splits=5)

    print(f"Classification Accuracy: {mean_acc}, \n Standard Deviation: {std_acc}")
    print(f"Confusion matrix: \n{confusion_matrix}")
    disp = ConfusionMatrixDisplay(confusion_matrix, display_labels=None).plot(cmap="Blues")
    # disp.plot()
    # plt.show()
    print(df_results)

    # Inter/intra attribute distance
    print(distance_based_attribute_evaluation(X=embeddings.detach(), y=torch.nn.functional.one_hot(y_label).detach(), attribute_names=attribute_names))
    print("\n\n")

async def evaluation_pipeline():

    #----------- Comment if local
    parser = argparse.ArgumentParser(description='Generate a model from params and save it to cloud storage')

    parser.add_argument('--version', type=str, help='Model version. Default is "latest"', required=True)
    # parser.add_argument('--description', type=str, help='Model description', required=True)
    # parser.add_argument('--training_path', type=str, help="Path in cloud storage", required=True)
    parser.add_argument('--configuration_name', type=str, help='Name of the bigquery configuration target', required=True)
    parser.add_argument('--run_local', type=bool, help='Run locally or in docker', required=True, default=False)
    parser.add_argument('--local_data', type=bool, help='Use local data or query table', required=True, default=False)
    # parser.add_argument('--save_embedding', type=bool, help='Use local data or query table', required=False, default=False)

    args = parser.parse_args()
    version = args.version or "latest"

    print(args)
    print(args.run_local)
    """Fetch and create the configuration files"""
    if args.run_local == False:
        # make sure the folder exists, if not create it
        if not os.path.exists("/app/src/training/config_files"):
            os.makedirs("/app/src/training/config_files")
        folder_for_yamls = "/app/src/training/config_files"
    else:
        if not os.path.exists("config_files"):
            os.makedirs("config_files")
        folder_for_yamls = "config_files"

    read_experiment_config_from_bigquery(args.configuration_name, folder_for_yamls=folder_for_yamls)


    if args.run_local == False:
        """ Dockerized """
        conf_path = "/app/src/training/config_files/tmp_parameters.yaml"
    else:
        """ Local """
        conf_path = "config_files/tmp_parameters.yaml"

    with open(conf_path, 'r') as f:
        config = yaml.safe_load(f)

    if args.run_local == False:
        """ Dockerized """
        exp_conf_path = "/app/src/training/config_files/tmp_experiment_config.yaml"
    else:
        """ Local """
        exp_conf_path = "config_files/tmp_experiment_config.yaml"

    with open(exp_conf_path, 'r') as f:
        exp_config = yaml.safe_load(f)

    if args.run_local == False:
        service_account_file = "/app/secrets/syb-production-ai/service_account.json"
    else:
        service_account_file = "../../secrets/syb-production-ai/service_account.json"



    gcs_client = storage.Client(project=config["gcs_bucket"]["project"])

    model_loader = ModelLoader(gcs_client=gcs_client,
                           bucket=get_bucket(gcs_client, config))

    # Download models from GCS
    available_models = model_loader.list_available_models(prefix="models/")
    print(f"Available models are {available_models}")
    download_folder = model_loader.download_model(gcs_path=exp_config['gcs_path'],
                                model_name="DeepSet",
                                model_version="geomloss_quadratic-d256-2024-01-01", #"geomloss_quadratic-d256-2024-01-01",
                                download_folder="../../trained_models/")


    # Load local model
    local_model = model_loader.load_model(model_path="../../trained_models/models/",
                                          model_name="DeepSet",
                                          model_version="geomloss_quadratic-d256-2024-01-01")

    print(local_model.info)



    # Download the data set with labels
    if args.local_data == True:
        df = pd.read_json(path_or_buf="../../data/top_readymades_embedding.json")
        # df = pd.read_json(path_or_buf="../../data/user_playback_balanced_v3.json")
    else:
        # self.logger.info(f"Starting run {self.exp_config['name']} for partition {self.exp_config['partition_date']}")
        data_loader = DataLoader(config=config["config"]["bigquery"], service_account_file=service_account_file)

        df = await data_loader.load_data(partition_date=exp_config['partition_date'])

        # df.to_json(path_or_buf="../../data/user_playback_v0.json")


    preprocesser = Preprocesser(input_df=df, groupby_str="playlist_id")#config["config"]["bigquery"]["groupby_str"])
    dataset = preprocesser.preprocess()

    # self.logger.info(f"Preprocessed data has {len(dataset)} instances.")


    # Use the id strings in the dataset to get the attribute labels

    x, labels = zip(*dataset)

    """ Business type based attributes """
    # test_business_type_evaluation(df, local_model, dataset, exp_config)
    # return

    """ Top tags based attributes 
        _______________________________
    """

    # df_attributes = pd.read_json(path_or_buf="../../data/top_tags_readymades_user_playback_balanced.json")
    # label_ids, attribute_names, attribute_labels = get_attribute_labels(df_attributes, column_name="tag", groupby_str="account_id", are_grouped=False)
    df_attributes = pd.read_json(path_or_buf="../../data/airtable_readymade_genre_tags_substantial.json")

    # Take the rows of df_attributes where the playlist_id is in df["playlist_id".unique()
    df_attributes_subset = df_attributes.loc[df_attributes["playlist_id"].isin(df["playlist_id"].unique())]

    label_ids, attribute_names, attribute_labels = get_attribute_labels(df_attributes_subset, column_name="genre_tags_list", groupby_str="playlist_id", are_grouped=True)


    # Use the model
    model = local_model.model
    model.eval()

    print(f"num_params: {count_parameters(model)}")

    if len(dataset) > len(label_ids):
        print("WARNING: The dataset is larger than the attribute labels. Only doing the instances with labels.")
        embeddings = torch.zeros(size=(len(label_ids), exp_config[exp_config["model_type"]]["output_dim"]))
    else:
        embeddings = torch.zeros(size=(len(dataset), exp_config[exp_config["model_type"]]["output_dim"]))

    # Get the embeddings
    lengths = torch.zeros(size=(len(label_ids),))
    j = 0
    for i in range(len(dataset)):
        x_input, label = dataset[i]
        if label in label_ids:
            embeddings[j] = model(x_input.unsqueeze(0))
            lengths[j] = x_input.shape[0]
            j += 1

    #assert j == len(label_ids), "The number of instances with labels is not the same as the number of labels."


    """ Do the evaluation (classification & distance based) """
    # mean_acc, std_acc, confusion_matrix = classification_based_attribute_evalutation(X=embeddings.detach(), y_label=attributes.detach(), n_splits=5)
    df_results = multilabel_classification_based_attribute_evaluation(X=embeddings.detach(), y_label=attribute_labels, attribute_names=attribute_names, n_splits=5)

    pd.set_option('display.max_columns', 15)
    pd.set_option('display.width', 2500)
    print(df_results)


    embeddings_base = torch.zeros(size=(len(label_ids), 141))
    embeddings_base_2 = torch.zeros(size=(len(label_ids), 73))
    j = 0
    for i in range(len(dataset)):
        x_input, label = dataset[i]
        if label in label_ids:
            embeddings_base[j] = torch.mean(x_input, dim=0)
            embeddings_base_2[j] = get_genre_tag_from_track_set(x_input, fraction_top_tracks=0.5, only_genre=True)
            j += 1

    df_results_avg = multilabel_classification_based_attribute_evaluation(X=embeddings_base.detach(), y_label=attribute_labels, attribute_names=attribute_names, n_splits=5)
    df_results_topavg = multilabel_classification_based_attribute_evaluation(X=embeddings_base_2.detach(), y_label=attribute_labels, attribute_names=attribute_names, n_splits=5)

    print(df_results_topavg)
    print(df_results_avg)


    #
    # # Use the model
    # model = local_model.model
    # model.eval()
    #
    # embeddings = torch.zeros(size=(len(dataset), exp_config[exp_config["model_type"]]["output_dim"]))
    #
    # # Get the embeddings
    # lengths = torch.zeros(size=(len(dataset),))
    # for i in range(len(dataset)):
    #     x_input, label = dataset[i]
    #     embeddings[i] = model(x_input.unsqueeze(0))
    #     lengths[i] = x_input.shape[0]
    #
    # mask_10 = lengths > 10
    # mask_100 = lengths > 100
    # masked_embeddings_10 = embeddings[mask_10,:]
    # masked_embeddings_100 = embeddings[mask_100,:]
    #
    #
    # """ Save the embeddings """
    #
    # print(labels)
    # print(embeddings.shape)
    # print([attribute_names[attributes[i]] for i in range(len(attributes))])
    # print(attributes)
    # df_result = pd.DataFrame({"account_id": labels, "business_type": [attribute_names[attributes[i]] for i in range(len(attributes))], "embedding": list(embeddings.detach().numpy())})
    # df_result.to_json(path_or_buf="../../data/embeddingsV0_d128_4types.json")
    #
    # """ Do the evaluation (classification & distance based) """
    # # mean_acc, std_acc, confusion_matrix = classification_based_attribute_evalutation(X=embeddings.detach(), y_label=attributes.detach(), n_splits=5)
    # # df_results = multilabel_classification_based_attribute_evaluation(X=embeddings.detach(), y_label=torch.nn.functional.one_hot(attributes).detach(), attribute_names=attribute_names, n_splits=5)
    #
    # # select only the tensors from embeddings that are of length greater than 10, and remove the other tensors
    #
    # embeddings_list = [embeddings, masked_embeddings_10, masked_embeddings_100]
    # labels_list = [attributes, attributes[mask_10], attributes[mask_100]]
    # embeddings_names = ["All", "Length > 10", "Length > 100"]
    # for i in range(len(embeddings_list)):
    #
    #     embeddings = embeddings_list[i]
    #     y_label = labels_list[i]
    #     name = embeddings_names[i]
    #
    #     """sanity check: random y_label, same embeddings. """
    #     # y_label = torch.randint(0, 4, size=(len(y_label),))
    #
    #     print(f"Classification based evaluation: {name}")
    #     mean_acc, std_acc, confusion_matrix = classification_based_attribute_evalutation(X=embeddings.detach(), y_label=y_label.detach(), n_splits=5, id_labels=labels)
    #     df_results = multilabel_classification_based_attribute_evaluation(X=embeddings.detach(), y_label=torch.nn.functional.one_hot(y_label).detach(), attribute_names=attribute_names, n_splits=5)
    #
    #     print(f"Classification Accuracy: {mean_acc}, \n Standard Deviation: {std_acc}")
    #     print(f"Confusion matrix: \n{confusion_matrix}")
    #     disp = ConfusionMatrixDisplay(confusion_matrix, display_labels=None).plot(cmap="Blues")
    #     # disp.plot()
    #     # plt.show()
    #     print(df_results)
    #
    #     # Inter/intra attribute distance
    #     print(distance_based_attribute_evaluation(X=embeddings.detach(), y=torch.nn.functional.one_hot(y_label).detach(), attribute_names=attribute_names))
    #     pd.set_option('display.max_columns', 10)
    #     pd.set_option('display.width', 1500)
    #     print("\n\n")
    #
    #
    # """Spa ids that are classified as restaurant:
    # QWNjb3VudCwsMWI1aHk3aG4zc3cv
    # QWNjb3VudCwsMWFicGZwaTdveHMv
    # QWNjb3VudCwsMWV2czJ5czkzd2cv
    # QWNjb3VudCwsMWp3eGdmOWVwejQv
    # QWNjb3VudCwsMWF6aGxkdTdkMzQv
    # QWNjb3VudCwsMWMzcGNibG5pdGMv
    # QWNjb3VudCwsMWpmM3V4aThiMjgv
    # QWNjb3VudCwsMTluazg2Z29wb2cv
    # QWNjb3VudCwsMW9ib3V6cjZtdGMv
    # """
    # df_one_acc = df.loc[df['account_id'] == "QWNjb3VudCwsMWI1aHk3aG4zc3cv"]
    # print(df_one_acc["uri"].head(100))
    #
    # df_one_acc = df.loc[df['account_id'] == "QWNjb3VudCwsMWFicGZwaTdveHMv"]
    # print(df_one_acc["uri"].head(100))
    # #
    # # mean_acc, std_acc, confusion_matrix = classification_based_attribute_evalutation(X=embeddings.detach(), y_label=attributes.detach(), n_splits=5)
    # # df_results = multilabel_classification_based_attribute_evaluation(X=embeddings.detach(), y_label=torch.nn.functional.one_hot(attributes).detach(), attribute_names=attribute_names, n_splits=5)
    # #
    # # print("Classification based evaluation: All")
    # # print(f"Classification Accuracy: {mean_acc}, \n Standard Deviation: {std_acc}")
    # # print(f"Confusion matrix: \n{confusion_matrix}")
    # # disp = ConfusionMatrixDisplay(confusion_matrix, display_labels=None).plot(cmap="Blues")
    # # disp.plot()
    # # plt.show()
    # # print(df_results)
    #
    #
    #
    #





def main():
    # test_evaluation_gaussian()
    # test_evaluation_classifier_gaussian()
    # asyncio.run(evaluation_pipeline())
    asyncio.run(evaluation_pipe())

if __name__ == "__main__":
    main()