
import pandas as pd
import torch

from src.other.nearest_neighbors import get_n_neighbors


def jaccard_index(tags1, tags2):
    """ Jaccard index for tag set similarity. Done only by string matching.
    :param tags1: A list of tags, as strings.
    :param tags2: A list of tags, as strings.
    :return: The Jaccard index, as a float.
    """
    intersection = len(set(tags1).intersection(tags2))
    union = len(set(tags1).union(tags2))
    return intersection / union


""" From one list of tags, to a df with tags"""
def tags_to_df_similarity(tags, df, tag_type, groupby_str="playlist_id", type_column_name="category", tag_column_name="tags"):
    """
    :param tags: A list of tags to match with the df
    :param df: A dataframe with tags.
    :param tag_type: The type of tag that tags contain and which should be matched in the df.
    :param groupby_str: The column name that defines an instance, ie some form of id. Already assumed to be grouped.
    :param type_column_name: The name of the column in the df that contains the tag type.
    :param tag_column_name: The name of the column in the df that contains the tag name.
    :return:
    """
    # Find the df with only the relevant tags wrt to the tag_type
    df = df[df[type_column_name] == tag_type]

    # For each row in the df, calculate the jaccard index with the tags found under tag_column_name and the list tags.

    result = df.apply(lambda row: jaccard_index(tags, row[tag_column_name]), axis=1)

    # join with the groupby_str
    result = pd.concat([df[groupby_str], result, df[tag_column_name]], axis=1)
    # rename the column with the jaccard index
    result.rename(columns={0: f"jaccard_index_{tag_type}"}, inplace=True)

    # Return a list of labels and the values of the jaccard index.
    return result[f"jaccard_index_{tag_type}"].to_list(), result[groupby_str].to_list()


def p2p_embedding_tags_similarity(start_embedding, start_label, tags, embeddings, labels, df_tags, tag_type):
    """
    Will produce a dataframe with all the playlists in
    """
    dists = torch.cdist(start_embedding.unsqueeze(0), embeddings, p=2)

    # rank the embeddings by the distance to the start_embedding
    sorted_indices = torch.argsort(dists.squeeze(0))
    sorted_labels = [labels[i] for i in sorted_indices]

    jaccard_index, labels_with_jaccard = tags_to_df_similarity(tags, df_tags, tag_type)

    jaccard_df = pd.DataFrame({"playlist_id": labels_with_jaccard, f"jaccard_index_{tag_type}": jaccard_index})
    dist_df = pd.DataFrame({"playlist_id": sorted_labels, "distance": dists[0, sorted_indices].tolist()})

    result = pd.merge(dist_df, jaccard_df, on="playlist_id", how="left")

    # # Check if start_label is in the result df
    # if not start_label in result["playlist_id"].to_list():
    #     # Add it to the result df
    #     result.loc[-1] = [start_label, 0., 1.]
    #     result.index = result.index + 1
    #     result.sort_index(inplace=True)

    return result




import argparse
from src.training.div.model_utils import load_model_from_gcs
from src.training.data_handling.data_utils import extract_sets
from src.training.div.utils import get_readymade_names

def main():
    #----------- Comment if local
    parser = argparse.ArgumentParser(description='Generate a model from params and save it to cloud storage')

    parser.add_argument('--version', type=str, help='Model version. Default is "latest"', required=True)
    # parser.add_argument('--description', type=str, help='Model description', required=True)
    # parser.add_argument('--training_path', type=str, help="Path in cloud storage", required=True)
    parser.add_argument('--configuration_name', type=str, help='Name of the bigquery configuration target', required=True)
    parser.add_argument('--run_local', action='store_true', help='Run locally or in docker')
    parser.set_defaults(run_local=False)
    # parser.add_argument('--local_data', type=bool, help='Use local data or query table', required=True, default=False)
    # parser.add_argument('--save_embedding', type=bool, help='Use local data or query table', required=False, default=False)

    args = parser.parse_args()
    version = args.version or "latest"

    """ Load the data"""
    playlist_df = pd.read_json("../../data/top_readymades_embedding.json")

    """ Load model from GCS """
    model_version = "geomloss_quadratic-2024-01-01"
    model_name = "SetTransformer"
    model = load_model_from_gcs(args, model_version, model_name)


    """ Load the playlist dataset and embed"""
    dataset = extract_sets(playlist_df, groupby_str="playlist_id")
    (x, labels) = zip(*dataset)

    # Embed
    only_genre = False
    """ We can't really set it like this though? Possibly if we train the model with different types of masks. Genre-only, etc."""

    embeddings_list = []
    for i in range(len(x)):
        x_input = x[i].unsqueeze(0)
        if only_genre:
            # create mask to set all values to zero except the genre values
            mask = torch.zeros_like(x_input)
            mask[:, :, 31:31+73] = 1.
            x_input = x_input * mask
        embeddings_list.append(model(x_input).detach().squeeze(0))
    embeddings = torch.stack(embeddings_list)

    """ Load the playlist tags. """
    df_tags = pd.read_json("../../data/top_readymades_airtable_tags.json")
    id_name_dict = get_readymade_names()

    #------------------------------------------------
    """ Load the user embeddings. """
    df_users = pd.read_json("../../data/user_playback_balanced_v3.json")
    dataset_users = extract_sets(df_users, groupby_str="account_id")
    (x_users, labels_users) = zip(*dataset_users)

    # Embed the users
    embeddings_users_list = []
    for i in range(len(x_users)):
        x_input = x_users[i].unsqueeze(0)
        if only_genre:
            # create mask to set all values to zero except the genre values
            mask = torch.zeros_like(x_input)
            mask[:, :, 31:31+73] = 1.
            x_input = x_input * mask
        embeddings_users_list.append(model(x_input).detach().squeeze(0))
    embeddings_users = torch.stack(embeddings_users_list)


    # Load the user tags
    df_users_tags = pd.read_json("../../data/user_tags_taste_profile.json")

    #------------------------------------------------


    tag_type_list = ["genres", "sounds", "decades"]
    tag_type = "genres"
    # playlist_of_interest = ["Q29sbGVjdGlvbiwsMThwaGJkbjRsYzAvQ29tcG9zZXIsY3VyYXRvci1taXhlci1jb21wb3NlciwwLw..",
    #                         "Q29sbGVjdGlvbiwsMWVkYmFiMXRmY3cvQ29tcG9zZXIsY3VyYXRvci1taXhlci1jb21wb3NlciwwLw.."]
    playlist_of_interest = []

    for i in range(len(embeddings_users)):
        result_dfs = []

        start_embedding = embeddings_users[i]
        start_label = labels_users[i]
        _tmp_df = df_users_tags[df_users_tags["account_id"] == start_label]

        if len(_tmp_df.index) == 0:
            """ We don't have any tags for this user"""
            continue

        for tag_type in tag_type_list:
            tags = _tmp_df[_tmp_df["category"] == tag_type]["tags"].to_list()
            if len(tags) == 0:
                """ The user doesn't have any tags of this type."""
                continue

            tags = tags[0]

            tag_type_result = p2p_embedding_tags_similarity(start_embedding, start_label, tags, embeddings, labels, df_tags, tag_type)

            result_dfs.append(tag_type_result)

        print(f"Results for start label: {start_label}")
        df_results = pd.concat(result_dfs, axis=1)
        df_results = df_results.loc[:,~df_results.columns.duplicated()].copy()

        # Add the playlist name as the second column
        df_results.insert(1, "playlist_name", [id_name_dict[id_str] for id_str in df_results["playlist_id"]])

        print(df_results)


        input("Press Enter to continue...")







    df = pd.read_json("../../data/top_readymades_airtable_tags.json")
    labels = df["playlist_id"].to_list()

    d = len(df.index)
    start_embedding = torch.randn(size=(d,))
    embeddings = torch.randn(size=(len(labels), d))
    start_label = "alskdjföawe"




    p2p_embedding_tags_similarity(start_embedding,start_label, tags, embeddings, labels, df, tag_type="genres")



    # tags1 = []
    # tags2 = ["b"]
    # print(f"test JI: {jaccard_index([], ['b'])}")
    # # print(f"test JI: {jaccard_index([], [])}")
    #
    #
    #
    # df = pd.read_json("../../data/top_readymades_airtable_tags.json")
    # df_1 = tags_to_df_similarity(["latest"], df, "decades")
    # df_2 = tags_to_df_similarity(["pop"], df, "genres")
    # df_3 = tags_to_df_similarity(["happy"], df, "sounds")
    #
    # _tmp = pd.merge(df_1, df_2, on="playlist_id", how="outer")
    # result = pd.merge(_tmp, df_3, on="playlist_id", how="outer")
    # print(result)



if __name__ == "__main__":
    pd.set_option('display.max_columns', None, 'display.max_rows', 500, 'display.width', 2000, 'display.max_colwidth', 1000)
    main()