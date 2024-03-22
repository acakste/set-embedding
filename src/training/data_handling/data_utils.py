import torch
import pandas as pd
import os
from google.cloud import bigquery

def collate_fn_pad_seqs(batch):
    """ Assume x_batch """
    (xs, labels) = zip(*batch)

    # Get sequence lengths
    lengths = torch.tensor([t.shape[0] for t in xs])
    # Pad
    x_batch_padded = torch.nn.utils.rnn.pad_sequence(xs,batch_first=False, padding_value=0.0)

    # Compute mask
    mask = (x_batch_padded != 0)
    return x_batch_padded, labels, lengths, mask

def extract_sets(df, groupby_str="account_id"):
    """ Expecting df with columns ['{groupby_str}', 'vector']
        Groups the rows with same '{groupby_str}', creates a set of #rows vectors.
    """
    df_group = df.groupby([groupby_str])

    # Create a list of tensors, each list is a sequence of vector embedding elements
    data = []
    for key, item in df_group:
        seq_df = df_group.get_group(key[0])

        _tmp = pd.DataFrame(seq_df["vector"].to_list())
        data.append([torch.tensor(_tmp.values, dtype=torch.float32, requires_grad=True).contiguous(), str(key[0])])

    return data


def get_unique_embeddings(df):
    df_unique_uri = df.drop_duplicates(subset="uri")
    collapsed_vectors = pd.DataFrame(df_unique_uri["vector"].to_list())
    uris = df_unique_uri["uri"].to_list()
    return collapsed_vectors.to_numpy(), uris


def read_tags(file_name, file_path="../data/"):
    df = pd.read_json(path_or_buf=file_path + file_name + ".json")
    print(df)
    df['playlist_id'] = df['playlist_id'].astype(str)
    return df
def read_query_file(file_name, file_path="../query/"):
    with open(file_path + file_name + ".txt") as f:
        lines = f.read()
    return lines

def get_embeddings(embedding_query, project="syb-production-ai", format="json"):
    """CSV doesn't work atm. """
    query_data_file = f"../../data/{embedding_query}.{format}"

    if not os.path.exists(query_data_file):
        """ No csv file with query name exists. Query and write to csv file. """
        bigquery_client = bigquery.Client(project=project)
        test_query = read_query_file(embedding_query)
        query_job = bigquery_client.query(query=test_query)
        result_df = query_job.to_dataframe()
        print(result_df)
        print(result_df.columns)
        if format == "json":
            result_df.to_json(path_or_buf=query_data_file)
        elif format == "csv":
            result_df.to_csv(path_or_buf=query_data_file, columns=result_df.columns)#,sep="\t", index=False, quotechar="|", quoting=csv.QUOTE_ALL)
        read_result_df = pd.read_csv(filepath_or_buffer=query_data_file)
        print(read_result_df)
        print(read_result_df.columns)

    else:
        """ Json file with query name exists. Use data from that file. """
        print(f"Data with name {embedding_query} exists, using that.")
        if format == "json":
            result_df = pd.read_json(path_or_buf=query_data_file)
        elif format == "csv":
            raise NotImplementedError
        else:
            raise ValueError(f"Format {format} not supported.")

    embeddings = pd.DataFrame({"playlist_id": result_df["playlist_id"],
                               "uri": result_df["uri"],
                               "vector": result_df["vector"].to_list()})
    return embeddings
