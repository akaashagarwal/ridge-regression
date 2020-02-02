"""Module to load and preprocess data."""
from typing import List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataWrangling():
    """Class for data loading and preprocessing.

    Specifically, it provides the following functionalities:
        1. Load data from a .csv or .excel source.
        2. Split the data into training and test sets accoriding to given proportions.
        3. Standardize features of the training set.
        4. Use means and variances of the training data to standardize the test set.

    Attributes:
        data_path (str): Path to data file.

    """

    def __init__(self, data_path: str = '') -> None:
        """Initialise.

        Args:
            data_path (str, optional): File path of data file.

        """
        self.data_path = data_path
        self._data = pd.DataFrame()
        self._train_means: np.array
        self._train_vars: np.array

    @property
    def data(self) -> pd.DataFrame:
        """Return deep copy of self._data.

        Returns:
            pd.DataFrame

        """
        return self._data.copy(deep=True)

    @data.setter
    def data(self, data) -> None:
        """Setter method for attribute self._data."""
        self._data = data

    @property
    def train_means(self) -> np.array:
        """Return self._train_means."""
        return self._train_means

    @property
    def train_vars(self) -> np.array:
        """Return self._train_vars."""
        return self._train_vars

    def load_data(self) -> None:
        """Read data from file given by self.data_path."""
        if self.data_path.endswith('.csv'):
            self._data = pd.read_csv(self.data_path)
        elif self.data_path.endswith('.xlsx'):
            self._data = pd.read_excel(self.data_path)
        self._data.columns = list(map(str.lower, self._data.columns))  # Lower case of all cols.

    def store_train_means_vars(self, scaler) -> None:
        """Assign mean and variance to class attributes."""
        self._train_means = scaler.mean_
        self._train_vars = scaler.var_

    def standardize_features(self, dframe: pd.DataFrame, features: List[str]) -> np.array:
        """Standardize features to zero mean and unit variance.

        Args:
            dframe (pd.DataFrame): Input design matrix.
            features (List[str]): List of feature names which are columns of dframe.

        Returns:
            np.array: Standardized input data frame.

        """
        scaler = StandardScaler()
        scaler.fit(X=dframe[features])
        self.store_train_means_vars(scaler)
        return scaler.transform(X=dframe[features])

    def standardize_test_set(self, dframe: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """Use training set feature means and variances to normalize test set.

        Args:
            dframe (pd.DataFrame): Test set.
            features (List[str]): Test feature column names.

        Returns:
            pd.DataFrame: Standardized test set using training data characteristics.

        """
        centered_df = dframe[features] - self._train_means
        scaled_df = centered_df / (self._train_vars**0.5)
        return scaled_df

    @staticmethod
    def one_hot_encode_features(dframe: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """Encode categorical variables by using the one-hot encoding technique.

        Args:
            dframe (pd.DataFrame): Input data.
            features (List[str]): List of categorical feature column names.

        Returns:
            pd.DataFrame: One hot encoded data frame.

        """
        one_hot_encoded_df = pd.DataFrame()
        for feature in features:
            ohe_feature = pd.get_dummies(data=dframe[feature], prefix=feature + '_')
            one_hot_encoded_df = pd.concat((one_hot_encoded_df, ohe_feature), axis=1)
        return one_hot_encoded_df

    @staticmethod
    def split_train_test(x_data: np.array, y_labels: np.array, features: list,
                         test_size: float) -> dict:
        """Split into training and test sets by proportion given by test_size.

        Returns:
            dict: Dictionary with training data, training labels, test data,
                  and test labels.

        """
        x_train, x_test, y_train, y_test = train_test_split(x_data,
                                                            y_labels,
                                                            test_size=test_size,
                                                            random_state=41,
                                                            shuffle=True)
        return_val = {
            'x_train': pd.DataFrame(x_train, columns=features),
            'x_test': pd.DataFrame(x_test, columns=features),
            'y_train': pd.Series(y_train),
            'y_test': pd.Series(y_test)
        }
        return return_val
