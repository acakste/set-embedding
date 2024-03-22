

import torch
import pandas as pd
import hdbscan
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import mplcursors


# UMAP
import umap
import umap.plot

from src.other.nearest_neighbors import get_n_neighbors

def cluster_embedding(user_embeddings, min_cluster_size=6, cluster_selection_epsilon=1e-2):

    # load embeddings from the embedding column of df
    # user_embeddings = torch.tensor(pd.DataFrame(df["embedding"].to_list()).to_numpy())

    """ HDBSCAN"""
    print(f"Fitting Hdbscan..")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, cluster_selection_epsilon=cluster_selection_epsilon, gen_min_span_tree=False, prediction_data=True)
    clusterer.fit(user_embeddings)

    mask_nocluster = clusterer.labels_ == -1
    sc_hdscan = silhouette_score(user_embeddings[~mask_nocluster], clusterer.labels_[~mask_nocluster])
    print(f"SC for HDSCAN (only clustered instances): {sc_hdscan}")
    print(f"Clustered {np.sum(mask_nocluster)}/{len(user_embeddings)} instances")
    print(clusterer.labels_)
    num_clusters_found = max(clusterer.labels_) + 1

    print(f"num_clusters_found: {num_clusters_found}")
    if num_clusters_found >= 1:
        # print(f"histogram of labels: {np.histogram(clusterer.labels_, bins=num_clusters_found)}")

        # create dictionary with the cluster labels as keys and the number of tracks in each cluster as values
        cluster_dict = {}
        for i in range(num_clusters_found):
            mask = clusterer.labels_ == i
            cluster_dict[i] = np.sum(mask)

        print(f"Cluster dict: {cluster_dict}")


    print(clusterer.probabilities_[~mask_nocluster])


    # test_labels, strengths = hdbscan.approximate_predict(clusterer, user_embeddings[mask_nocluster])

    # hdbscan_labels, kmeans_labels_ = clustering_analysis(user_embeddings, only_genre=False, min_cluster_size=6, cluster_selection_epsilon=1e-2)

    return clusterer


def show_info(sel, ids):
    """ Show the account_id and business_type when hovering over a point"""
    ind = sel.target.index
    if ids == None:
        sel.annotation.set_text(f'hey {ind}')
    else:
        sel.annotation.set_text(f'{ids[ind]}')

def reduce_user_embeddings(user_embeddings, labels, n_neighbors, min_dist, mapper=None):
    """ Reduce with UMAP, and plot the embeddings with colors from cluster labels"""
    if mapper == None:
        mapper = umap.UMAP(n_neighbors=n_neighbors, n_components=2, min_dist=min_dist).fit(user_embeddings)

    ax = umap.plot.points(mapper, labels=labels, width=1600, height=1600)
    # umap.plot.connectivity(mapper, labels=labels)
    # matplotlib.use('TkAgg')
    # Add hover info
    # mplcursors.cursor(hover=True).connect("add", show_info)

    plt.show()

    return mapper

def main():
    """ Cluster embeddings """
    df = pd.read_json("../../data/embeddings_d128_4types.json")
    account_ids = df["account_id"].to_list()
    business_types = df["business_type"].to_list()

    print(df)
    user_embeddings = torch.tensor(pd.DataFrame(df["embedding"].to_list()).to_numpy())
    clusterer = cluster_embedding(user_embeddings)


    """Business type labels"""
    attributes = list(df["business_type"].unique())
    print(attributes)
    # Get the indices of the business types


    """Top tag labels"""
    df_taste_profile_tags = pd.read_json("../../data/user_tags_taste_profile.json")



    # Get the tags for each account_id
    user_top_genre = []
    tag_type = "genres"
    for id in account_ids:
        _tmp_df = df_taste_profile_tags[df_taste_profile_tags["account_id"] == id]

        if len(_tmp_df.index) == 0:
            """ We don't have any tags for this user"""
            user_top_genre.append("None")
            continue
        tags = _tmp_df[_tmp_df["category"] == tag_type]["tags"].to_list()
        if len(tags) == 0:
            """ The user doesn't have any tags of this type."""
            user_top_genre.append("None")
            continue
        else:
            """ Only take certain tags for now."""
            # tags_of_interest = ["mainstream", "modern", "happy", "electronic", "indie", "acoustic", "retro"]
            tags_of_interest = ["pop", "pop-rock", "edm", "rock", "hip-hop", "jazz", "ambient"]
            # if tags[0][0] in tags_of_interest:
            #     user_top_genre.append(tags[0][0])
            # else:
            #     user_top_genre.append("other")
            if tags[0][0] == "pop":
                if len(tags[0]) == 1:
                    user_top_genre.append("pop")
                elif tags[0][1] in tags_of_interest:
                    user_top_genre.append(tags[0][1])
                else:
                    user_top_genre.append("pop_other")
            else:
                user_top_genre.append("other")
            # user_top_genre.append(tags[0][0])



    # Get the unique top genres present in the list user_top_genre
    top_genres_set = list(set(user_top_genre))
    top_genre_labels = np.array([top_genres_set.index(genre) for genre in user_top_genre])
    # count the number of types each label occurs in top_genre_labels
    print(f"top_genre_labels: {top_genre_labels}")
    from collections import Counter
    print(Counter(top_genre_labels))



    print(f"top_genres_set (used for cluster ids): {top_genres_set}")


    l = clusterer.labels_

    n_neighbors = 50
    mapper = reduce_user_embeddings(user_embeddings, clusterer.labels_, n_neighbors=n_neighbors, min_dist=0.)
    reduce_user_embeddings(user_embeddings, np.array(business_types), n_neighbors=n_neighbors, min_dist=0., mapper=mapper)
    reduce_user_embeddings(user_embeddings, np.array(user_top_genre), n_neighbors=n_neighbors, min_dist=0., mapper=mapper)



    """ For each user, get the 20 nearest neighbors and print them along clusters and distances. (Not relevant anymore, better with distance_to_tag_order.py)"""
    # neighbor_dists, neighbor_indices = get_n_neighbors(user_embeddings, 20)
    # for i in range(len(account_ids)):
    #     _ids = []
    #     _bt = []
    #     _dists = []
    #     _cluster = []
    #     for j in range(len(neighbor_indices[i])):
    #         _ids.append(account_ids[neighbor_indices[i][j]])
    #         _bt.append(business_types[neighbor_indices[i][j]])
    #         _dists.append(neighbor_dists[i][j])
    #         _cluster.append(clusterer.labels_[neighbor_indices[i][j]])
    #
    #     df = pd.DataFrame({"account_id": _ids, "business_type": _bt, "cluster": _cluster, "dist": _dists})
    #     # Sort df on neighbor_dist
    #     df = df.sort_values(by=["dist"])
    #     # if clusterer.labels_[neighbor_indices[i][0]] == 4 and clusterer.labels_[neighbor_indices[i][1]] == 5:
    #     print(df)
    #     input("Press Enter to continue...")

if __name__ == "__main__":
    main()