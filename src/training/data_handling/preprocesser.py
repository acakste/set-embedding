import pandas as pd

from src.training.data_handling.data_utils import extract_sets

class Preprocesser:
    def __init__(self, input_df: pd.DataFrame, groupby_str: str):
        self.input_df = input_df
        self.groupby_str = groupby_str

    def preprocess(self):
        """ Expecting input_df to include (not exclusively) the following columns: ['{groupby_str}', 'vector']"""
        return extract_sets(self.input_df, groupby_str=self.groupby_str)


def get_attribute_labels(df, labels, groupby_str, attribute_str):
    attribute_labels = []
    for label in labels:
        attribute_series = df.loc[df[groupby_str] == label, attribute_str]

        # Check that every column value for each label is the same, if not raise an error
        if attribute_series.nunique() != 1:
            raise ValueError("The business type for the account_id is not the same for all the rows")

        # Get the unique business type
        attribute = attribute_series.unique()[0]
        attribute_labels.append(attribute)

    return attribute_labels
