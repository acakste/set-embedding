
import torch
import pandas as pd
import argparse

from src.training.data_handling.data_utils import extract_sets
from src.training.div.model_utils import load_model_from_gcs
from src.training.div.utils import get_readymade_names

"""Embed a set of playlists and a set of users, separately"""
def embed_set(dataset, model):
    embeddings = []
    for i in range(len(dataset)):
        x, _ = dataset[i]
        embeddings.append(model(x.unsqueeze(0)))

    return torch.stack(embeddings)

def embed_playlist_and_user(playlist_df, user_df, model):
    playlist_dataset = extract_sets(playlist_df, groupby_str="playlist_id")
    playlist_embeddings = embed_set(playlist_dataset, model)

    user_dataset = extract_sets(user_df, groupby_str="account_id")
    user_embeddings = embed_set(user_dataset, model)

    _, playlist_ids = zip(*playlist_dataset)
    _, user_ids = zip(*user_dataset)
    return (playlist_embeddings, playlist_ids), (user_embeddings, user_ids)

def similarity_search(origin, embeddings, k=10):
    """Find the most similar embeddings to the single point origin"""
    dist = torch.cdist(origin, embeddings).squeeze(-1)
    values, indices = torch.topk(-dist, k, dim=0)
    return indices, -values


"""Search for the most similar playlist for a user"""
def most_similar_playlist_for_user(playlist_df, user_df, model, k=10):

    (playlist_embeddings, playlist_ids), (user_embeddings, user_ids) = embed_playlist_and_user(playlist_df, user_df, model)

    id_name_mapping = get_readymade_names()
    result = []
    for i in range(len(user_embeddings)):
        # Get the most similar playlists for the user
        indices, values = similarity_search(user_embeddings[i], playlist_embeddings, k)

        # Unpack the playlist_ids and names
        similar_playlists_ids = [playlist_ids[int(id)] for id in indices]
        similar_playlists_names = [id_name_mapping[id_str] for id_str in similar_playlists_ids]

        # create a dataframe with the results
        df = pd.DataFrame({"account_id": k*[str(user_ids[i])], "distance": list(values.detach()), "playlist_id": similar_playlists_ids, "playlist_name": similar_playlists_names})
        result.append(df)
    return result


"""Search for the most similar user for a playlist"""


def main():
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

    """ Load the data"""
    playlist_df = pd.read_json("../../data/top_readymades_embedding.json")
    user_df = pd.read_json("../../data/user_playback_balanced_v3.json")

    """ Load model from GCS """
    model_version = "geomloss_quadratic-d256-2024-01-01"
    model_name = "SetTransformer"
    model = load_model_from_gcs(args, model_version, model_name)

    """ Search for the most similar playlist for a user"""
    result = most_similar_playlist_for_user(playlist_df, user_df, model, k=40)

    pd.set_option('display.max_rows', None, 'display.max_columns', None, 'display.width', None)
    for i in range(len(result)):
        print(result[i])
        input("Press Enter to continue...")

if __name__ == "__main__":
    main()