"""Module containing main driver function to load and preprocess the data, and make predictions."""
import pandas as pd
from src.algorithm import get_ridge_regression_score, get_sklearn_prediction_score
from src.datawrangling import DataWrangling

DATA_PATH = "src/data/data.xlsx"


def preprocess_data(dframe: pd.DataFrame, datawrang_obj: DataWrangling) -> dict:
    """Preprocess data by using one-hot encoding, feature standardization, and splitting.

    Specfically, this function will one-hot encode the categorical variable 'rad'. Further, it will
    also split the data into training and test sets. Finally, it will standardize all features for
    all values to fall between the interval [0, 1].

    Args:
        dframe (pd.DataFrame): Input data frame containing samples with features and response data.
        datawrang_obj (DataWrangling): Object of class DataWrangling to aid in preprocessing data.

    Returns:
        dict: Dictionary with key:value pairs for training data, training labels, test data, and
            test labels.

    """
    class_col = 'mdv'
    continuous_feats = [
        "crim", "zn", "indus", "nox", "age", "dis", "tax", "ptratio", "b", "lstat", "rm"
    ]
    categorical_feats = ['rad']
    test_size = 0.2

    ohe_feature_df = datawrang_obj.one_hot_encode_features(dframe=dframe,
                                                           features=categorical_feats)
    dframe.drop(columns=categorical_feats, inplace=True)
    dframe = pd.concat([dframe, ohe_feature_df], axis=1)

    feature_list = list(dframe.columns)
    feature_list.remove(class_col)
    data_splits = datawrang_obj.split_train_test(x_data=dframe[feature_list].to_numpy(),
                                                 y_labels=dframe[class_col].to_numpy(),
                                                 features=feature_list,
                                                 test_size=test_size)

    data_splits['x_train'] = datawrang_obj.standardize_features(dframe=data_splits['x_train'],
                                                                features=continuous_feats)
    data_splits['x_test'] = datawrang_obj.standardize_test_set(dframe=data_splits['x_test'],
                                                               features=continuous_feats)

    return data_splits


def predict() -> tuple:
    """Run prediction method and return MAPE scores of both custom and sklearn regression models.

    Returns:
        tuple: Mean absolute percentage error (MAPE) of custom model, and MAPE of sklearn model on
            same test data.

    """
    datawrang_obj = DataWrangling(data_path=DATA_PATH)
    datawrang_obj.load_data()

    data_splits = preprocess_data(dframe=datawrang_obj.data, datawrang_obj=datawrang_obj)

    ridge_reg_score = get_ridge_regression_score(data_splits, reg_strength=10, max_iter=1000)
    sklearn_score = get_sklearn_prediction_score(data_splits, reg_strength=10, max_iter=1000)

    return ridge_reg_score, sklearn_score
