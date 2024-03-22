import torch
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support, average_precision_score, precision_score
import numpy as np
import matplotlib.pyplot as plt
import sys

def distance_based_attribute_evaluation(X, y, attribute_names):
    """
    This function evaluates the distance between instances sharing the attribute and instances not sharing the attribute.
    :param X: The embedded data instances in matrix form. (N x d)
    :param y: The binary attribute labels (N x len(attribute_names)) where the labels are {0,1}
    :param attribute_names: The attribute names
    :return:

    Maybe nice to make it possible for attributes names to be classes. How the distance between the classes are instead of
    just positive and negative attribute class. Probably a separate function for that.
    """
    # Each attribute defines two classes, the positive and the negative class.

    # Report the intra-class and inter-class average pairwise distance. As well as the centroid distance between the positive and negative class.


    # Compute all pairwise distances between the instances.
    pairwise_distances = torch.cdist(X, X, p=2)
    N_X, _ = X.shape
    N_y, n_attributed = y.shape

    assert N_X == N_y, "The number of instances in X and y must be the same."
    assert n_attributed == len(attribute_names), "The number of attributes in y must be the same as the number of attribute names."

    # Create a dataframe to store the results.
    df_results = pd.DataFrame(columns=["attribute", "pos_mean_dist", "neg_mean_dist", "inter_mean_dist", "pos_std_dist", "neg_std_dist", "inter_std_dist"])


    for i in range(len(attribute_names)):
        # Create a matrix of the same size as the pairwise_distances matrix, where the value is 1 if both are in the same positive class.
        # and 0 if they are not.
        positive_class_matrix_mask = (y[:,i].unsqueeze(1) @ y[:,i].unsqueeze(0))
        negative_class_matrix_mask = (1-y[:,i].unsqueeze(1)) @ (1-y[:,i].unsqueeze(0))
        inter_class_matrix_mask = (1-y[:,i].unsqueeze(1)) @ y[:,i].unsqueeze(0) + (y[:,i].unsqueeze(1)) @ (1-y[:,i].unsqueeze(0))

        # Choose the elements of the upper triangular matrix of pairwise_distances according to the masks.
        triu_indices = torch.triu_indices(N_X, N_X, offset=1)

        positive_class_pairwise_distances = torch.masked_select(pairwise_distances[triu_indices.unbind()], positive_class_matrix_mask.bool()[triu_indices.unbind()])
        negative_class_pairwise_distances = torch.masked_select(pairwise_distances[triu_indices.unbind()], negative_class_matrix_mask.bool()[triu_indices.unbind()])
        inter_class_pairwise_distances = torch.masked_select(pairwise_distances[triu_indices.unbind()], inter_class_matrix_mask.bool()[triu_indices.unbind()])


        avg_positive_distance = torch.mean(positive_class_pairwise_distances)
        avg_negative_distance = torch.mean(negative_class_pairwise_distances)
        avg_inter_distance = torch.mean(inter_class_pairwise_distances)
        std_positive_distance = torch.std(positive_class_pairwise_distances)
        std_negative_distance = torch.std(negative_class_pairwise_distances)
        std_inter_distance = torch.std(inter_class_pairwise_distances)

        # Add the values as a row in the dataframe.
        df_results.loc[i] = [attribute_names[i], avg_positive_distance, avg_negative_distance, avg_inter_distance, std_positive_distance, std_negative_distance, std_inter_distance]

    return df_results






def classification_based_attribute_evalutation(X, y_label, n_splits=10, random_state=42, id_labels=None, is_binary=False):
    """
    Fit a prediction model to predict the attribute.
    For hard labels, single class multi-class.
    :return:

    Based on SVC using K-fold cross validation.
    """

    """ Use sklearn SVC to do multi-class classification"""

    if is_binary and (sum(y_label) <= n_splits or sum(y_label) >= len(y_label) - n_splits):
        print(f"Too few positive/negative examples to do {n_splits} splits", file=sys.stderr)
        return {}, 1.

    # clf = make_pipeline(StandardScaler(), SVC(gamma='scale', kernel="rbf"))
    clf = make_pipeline(StandardScaler(), SVC(gamma='scale', kernel="rbf", class_weight="balanced", probability=True))
    # clf = make_pipeline(StandardScaler(), LogisticRegression(solver="liblinear", random_state=0))
    # Make k-fold stratification to fit and evaluate the model using SVC.
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    accuracies = []
    precisions = []
    recalls = []
    f_scores = []
    pr_curve_aps = []
    cms = []
    for i, (train_index, test_index) in enumerate(skf.split(X, y_label)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y_label[train_index], y_label[test_index]

        clf.fit(X_train, y_train)
        accuracies.append(clf.score(X_test, y_test))
        y_test_pred = clf.predict(X_test)
        cm = confusion_matrix(y_test, y_test_pred)
        cms.append(cm)

        if is_binary:
            precision, recall, f_score, _ = precision_recall_fscore_support(y_test, y_test_pred, average="binary", zero_division=np.nan)
            tmp = average_precision_score(y_test, clf.predict_proba(X_test)[:,1])
            precisions.append(precision)
            recalls.append(recall)
            f_scores.append(f_score)
        else:
            print("Implement multi-class precision, recall and f-score")
            pass

    metrics = {}
    # Support
    if is_binary:
        metrics["pos_support"] = int(sum(y_label))
        metrics["neg_support"] = int(len(y_label) - sum(y_label))

    # Accuracy
    metrics["accuracy_mean"] = np.mean(accuracies)
    metrics["accuracy_std"] = np.std(accuracies)

    # Precision
    metrics["precision_mean"] = np.mean(precisions)
    metrics["precision_std"] = np.std(precisions)
    if is_binary:
        metrics["precision_rand"] = 1. / (1 + (metrics["neg_support"] / metrics["pos_support"]))

    # Recall
    metrics["recall_mean"] = np.mean(recalls)
    metrics["recall_std"] = np.std(recalls)

    # F-score
    metrics["f_score_mean"] = np.mean(f_scores)
    metrics["f_score_std"] = np.std(f_scores)

    # Confusion matrix
    metrics["confusion_matrix"] = np.sum(cms, axis=0)

    # return np.sum(cms, axis=0), np.mean(accuracies), np.std(accuracies), np.mean(precisions), np.std(precisions), np.mean(recalls), np.std(recalls), np.mean(f_scores), np.std(f_scores)
    return metrics, 0.

def multilabel_classification_based_attribute_evaluation(X, y_label, attribute_names, n_splits=10, random_state=42):
    """
    :param X: the embeded vectors
    :param y_label: a vector of multilabels
    :param n_splits:
    :param random_state:
    :return:

    For each component of the multilabel vector, fit a binary classifier to predict the attribute.
    Subroutine classification_based_attribute_evalutation is based on SVC using K-fold cross validation.
    """

    # Check that input has correct dimensions.
    N_X, _ = X.shape
    N_y, n_attributed = y_label.shape

    assert N_X == N_y, "The number of instances in X and y must be the same."
    assert n_attributed == len(attribute_names), "The number of attributes in y must be the same as the number of attribute names."

    columns_defined = False

    for i in range(len(attribute_names)):
        y_binary_label = y_label[:,i]
        metrics, error_code = classification_based_attribute_evalutation(X, y_binary_label, n_splits=n_splits, random_state=random_state, is_binary=True)
        if error_code == 1.:
            """ There are too few positive/negative examples to do n_splits splits, skip the attribute."""
            continue

        if columns_defined == False:
            column_names = ["attribute"] + list(metrics.keys())
            df_results = pd.DataFrame(columns=column_names)
            columns_defined = True

        df_results.loc[i] = [attribute_names[i]] + [metrics[key] for key in column_names[1:]]

    return df_results



def get_attribute_labels(df, column_name, groupby_str="account_id", are_grouped=False):
    """ Create attribute labels from the df. The column_name indicates in which column the attribute values are located.
     The attribute values are assumed to be strings, and an id is assigned to each unique value.
     If grouped is set to True, the attributes are assumed to be in an array under the column_name.
     """

    if are_grouped == True:
        attribute_names = list(df[column_name].explode().unique())

    else:
        attribute_names = list(df[column_name].unique())
    label_ids = list(df[groupby_str].unique())

    attribute_labels = []
    for label in label_ids:

        # Get all attributes for the label_id
        if are_grouped == True:
            attribute_series = df.loc[df[groupby_str] == label, column_name].explode()
        else:
            attribute_series = df.loc[df[groupby_str] == label, column_name]

        # Get the unique attributes for the label_id
        attribute_for_label = attribute_series.unique()

        attribute = torch.zeros(size=(len(attribute_names),))
        for a in attribute_for_label:
            index = attribute_names.index(a)
            attribute[index] = 1.

        attribute_labels.append(attribute)

    return label_ids, attribute_names, torch.stack(attribute_labels)


def main():
    X = torch.randn(size=(1000, 10))
    y = torch.randint(0, 3, size=(1000,))
    attribute_names = ["A", "B", "C"]

    mean_acc, std_acc, cm = classification_based_attribute_evalutation(X, y)

    print(cm)
    # multilabel_classification_based_attribute_evaluation(X, torch.nn.functional.one_hot(y), attribute_names=attribute_names)

    # Make a pretty print with the confusion matrix and use the attribute names as labels.
    # print the confusion matrix with the attribute labels in attribute_names

    disp = ConfusionMatrixDisplay(cm, display_labels=attribute_names).plot(cmap="Blues")
    disp.plot()
    plt.show()






if __name__ == "__main__":
    main()