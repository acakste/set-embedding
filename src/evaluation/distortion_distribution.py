import warnings

from src.training.distortion.min_distortion import fn_quadratic
import torch
import matplotlib.pyplot as plt
import numpy as np

from geomloss import SamplesLoss




def sample_instance_pairs(N_train, N_test, sample_size):
    """

    :param N_train: An integer. The number of train examples.
    :param N_test: An integer. The number of test examples.
    :param sample_size: The number of instance pairs to sample.
    :return:
    Returns a tensor of dimension [sample_size, 2] for the training set and the test set.
    Each row contains the indices of the sampled instance pair.
    Contains no identity elements (i,i).
    """
    # Calculate the number of unique tuples in x_train, excluding identity tuples.
    num_tuples_train = int(N_train*(N_train-1)/2)
    num_tuples_test = int(N_test*(N_test-1)/2)

    # Sample a subset of the training and test set.
    train_indices = torch.randperm(num_tuples_train)[:sample_size]
    test_indices = torch.randperm(num_tuples_test)[:sample_size]

    if sample_size > num_tuples_train or sample_size > num_tuples_test:
        raise Warning(f"Warning: Sample size {sample_size} is larger than the number of unique tuples in the training set {num_tuples_train} or test set {num_tuples_test}.")

    # convert the indices to tuple indices.
    train_tuple_indices = torch.triu_indices(row=N_train, col=N_train, offset=1)
    test_tuple_indices = torch.triu_indices(row=N_test, col=N_test, offset=1)

    train_samples = train_tuple_indices[:,train_indices].transpose(0,1)
    test_samples = test_tuple_indices[:,test_indices].transpose(0,1)

    return train_samples, test_samples


def get_all_distortion_distribution(X, initial_dist_matrix, distortion_func=fn_quadratic, p=2, bins=10):
    """ X are the embedded instances. Computes the distribution for all pairs in X.
        Maybe add functionality to input pairs of interest?
    """
    N, d_embedd = X.shape

    embedding_dist_matrix = torch.cdist(X, X, p=p)

    assert embedding_dist_matrix.shape == initial_dist_matrix.shape

    triu_indices = torch.triu_indices(row=N, col=N, offset=1)

    embedding_dist_flattened = embedding_dist_matrix[triu_indices.unbind()]
    initial_dist_flattened = initial_dist_matrix[triu_indices.unbind()]

    distortion = distortion_func(embedding_dist_flattened, initial_dist_flattened, reduction="None")

    histogram, bin_number = torch.histogram(distortion, bins=bins)

    plt.hist(distortion, bins=10)
    plt.show()



def estimate_distortion_distribution(x_train, x_test, X_embedding_train, X_embedding_test, distortion_func, sample_size, max_num_tracks):

    N_train = len(x_train)
    assert N_train == X_embedding_train.shape[0]

    N_test = len(x_test)
    assert N_test == X_embedding_test.shape[0]

    train_distortion, test_distortion = train_test_distortion(x_train, x_test, X_embedding_train, X_embedding_test, distortion_func, sample_size, max_num_tracks)

    # Get the histogram for the train and test distortion.
    train_distortion = np.array(train_distortion)
    test_distortion = np.array(test_distortion)

    # Calculate histogram for train_distortion
    test_counts, test_bins = np.histogram(train_distortion)
    plt.hist(test_bins[:-1], test_bins, weights=test_counts, alpha=0.5, label="Train distortion")

    # calculate histogram for test_distortion
    train_counts, train_bins = np.histogram(test_distortion)
    plt.hist(train_bins[:-1], train_bins, weights=train_counts, alpha=0.5, label="Test distortion")

    plt.legend()
    # plt.hist(train_distortion, bins=10)
    plt.show()

    return np.mean(train_distortion), np.mean(test_distortion)


def _compute_distortion(x, X_embedding, distortion_func, sample_indices, max_num_tracks=None):
    """

    :param x: the input instances, a list of tensors [L_i, d_input] of length N
    :param X_embedding: The embedded tensors, a tensor [N, d_embed]
    :param distortion_func: a function pointer to the distortion function.
    :param sample_indices: A list of tuples that index pairs of x and X_embedding.
    :param max_num_tracks: Limit on the number of tracks to use for calculating the input distance.
    :return:
    """

    wasserstein_dist_fn = SamplesLoss(loss="sinkhorn", p=2, blur=0.05)

    distortion_list = []
    for i in range(len(sample_indices)):
        # Find the matrix indices for the sampled instance
        id_1, id_2 = sample_indices[i]

        if max_num_tracks != None:
            # Approximate the input dist by a random sample of the input.
            indices_A = torch.randperm(x[id_1].shape[0])[:max_num_tracks]
            indices_B = torch.randperm(x[id_2].shape[0])[:max_num_tracks]

            A = x[id_1][indices_A].contiguous()
            B = x[id_2][indices_B].contiguous()
        else:
            A = x[id_1].contiguous()
            B = x[id_2].contiguous()

        # Calculate the input and embedding dist.
        input_dist = wasserstein_dist_fn(A, B)
        embedding_dist = torch.cdist(X_embedding[id_1].unsqueeze(0), X_embedding[id_2].unsqueeze(0), p=2)

        # Note that mean reduction is only used to get a single float as return value. It is still a single instance pair.
        distortion_list.append(distortion_func(embedding_dist, input_dist, reduction="mean").detach())

    return distortion_list

def train_test_distortion(x_train, x_test, X_embedding_train, X_embedding_test, distortion_func, sample_size, max_num_tracks=None):
    """

    :param x_train: a list of input instances, where each instance is a tensor [L, d_input]
    :param x_test: a list of input instances, where each instance is a tensor [L, d_input]
    :param X_embedding_train: The embedding of the training instances, a tensor [N, d_embed]
    :param X_embedding_test: The embedding of the test instances, a tensor [N, d_embed]
    :param distortion_func: The distortion function used (in training). For example, fn_quadratic.
    :return:
    """

    N_train = len(x_train)
    assert N_train == X_embedding_train.shape[0]

    N_test = len(x_test)
    assert N_test == X_embedding_test.shape[0]

    # Get the indices for the sampled instance pairs.
    train_samples, test_samples = sample_instance_pairs(N_train=N_train, N_test=N_test, sample_size=sample_size)

    """ Compute the average distortion for the training set sample."""
    train_distortion = _compute_distortion(x_train, X_embedding_train, distortion_func, train_samples, max_num_tracks)
    test_distortion = _compute_distortion(x_test, X_embedding_test, distortion_func, test_samples, max_num_tracks)

    # for i in range(sample_size):
    #     # Find the matrix indices for the sampled instance
    #     id_1_train, id_2_train = train_samples[i]
    #
    #
    #     if max_num_tracks != None:
    #         # Approximate the input dist by a random sample of the input.
    #         indices_A = torch.randperm(x_train[id_1_train].shape[0])[:max_num_tracks]
    #         indices_B = torch.randperm(x_train[id_2_train].shape[0])[:max_num_tracks]
    #
    #         A = x_train[id_1_train][indices_A].contiguous()
    #         B = x_train[id_2_train][indices_B].contiguous()
    #     else:
    #         A = x_train[id_1_train].contiguous()
    #         B = x_train[id_2_train].contiguous()
    #
    #     # Calculate the input and embedding dist.
    #     input_dist = wasserstein_dist_fn(A, B)
    #     embedding_dist = torch.cdist(X_embedding_train[id_1_train].unsqueeze(0), X_embedding_train[id_2_train].unsqueeze(0), p=2)
    #
    #     # Note that mean reduction is only used to get a single float as return value. It is still a single instance pair.
    #     train_distortion.append(distortion_func(embedding_dist, input_dist, reduction="mean").detach())
    #
    # """ Compute the average distortion for the test set sample."""
    # test_distortion = []
    # for i in range(sample_size):
    #     # Find the matrix indices for the sampled instance
    #     id_1_test, id_2_test = test_samples[i]
    #
    #     # Calculate the input and embedding dist.
    #     input_dist = wasserstein_dist_fn(x_test[id_1_test].contiguous(),x_test[id_2_test].contiguous())
    #     embedding_dist = torch.cdist(X_embedding_test[id_1_test].unsqueeze(0), X_embedding_test[id_2_test].unsqueeze(0), p=2)
    #
    #     # Note that mean reduction is only used to get a single float as return value. It is still a single instance pair.
    #     test_distortion.append(distortion_func(embedding_dist, input_dist, reduction="mean").detach())



    return train_distortion, test_distortion

if __name__ == "__main__":

    sample_instance_pairs(1000, 300, 30)

    x_train = torch.randn(size=(1000, 100, 70))
    x_test = torch.randn(size=(300, 100, 70))

    X_embedding_train = torch.randn(size=(1000, 128))
    X_embedding_test = torch.randn(size=(300, 128))

    train_distortion, test_distortion = train_test_distortion(x_train, x_test, X_embedding_train, X_embedding_test, fn_quadratic, sample_size=30)

    print(f"Average Train distortion: {np.mean(train_distortion)}")
    print(f"Average Test distortion: {np.mean(test_distortion)}")

    estimate_distortion_distribution(x_train, x_test, X_embedding_train, X_embedding_test, fn_quadratic, sample_size=300)
    estimate_distortion_distribution(x_train, x_test, X_embedding_train, X_embedding_test, fn_quadratic, sample_size=300)
    estimate_distortion_distribution(x_train, x_test, X_embedding_train, X_embedding_test, fn_quadratic, sample_size=300)