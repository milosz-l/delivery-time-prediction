"""Python script to process the data"""

import datetime
from typing import List

import joblib
import numpy as np
import pandas as pd
from prefect import flow, task
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from config import Location, ProcessConfig


@task
def get_raw_data(data_locations: dict):
    """Read raw data

    Parameters
    ----------
    data_locations : dict
        Dictionary with the locations of the raw data
    """
    # fact table
    sessions_df = pd.read_json(data_locations["sessions_path"], lines=True)

    # dimension tables
    deliveries_df = pd.read_json(data_locations["deliveries_path"], lines=True)
    products_df = pd.read_json(data_locations["products_path"], lines=True)
    users_df = pd.read_json(data_locations["users_path"], lines=True)
    return sessions_df, deliveries_df, products_df, users_df


@task
def merge_data(
    sessions_df: pd.DataFrame,
    deliveries_df: pd.DataFrame,
    products_df: pd.DataFrame,
    users_df: pd.DataFrame,
    config: ProcessConfig,
):
    """Merge fact table and dimension tables

    Parameters
    ----------
    session_df : pd.DataFrame
        fact table
    deliveries_df : pd.DataFrame
        dimension table
    products_df : pd.DataFrame
        dimension table
    users_df : pd.DataFrame
        dimension table
    config : ProcessConfig
        config object with constants
    """
    # 1. Cut microseconds from `delivery_timestamp`, so it will be the same format as `purchase_timestamp`, because there are no microseconds in purchase_timestamp (using "." as a separator).
    deliveries_df["delivery_timestamp"] = deliveries_df["delivery_timestamp"].str.split(".", expand=True)[0]

    # 2. Change columns format to datetime.
    deliveries_df["purchase_timestamp"] = pd.to_datetime(deliveries_df["purchase_timestamp"], format=config.DATE_FORMAT)
    deliveries_df["delivery_timestamp"] = pd.to_datetime(deliveries_df["delivery_timestamp"], format=config.DATE_FORMAT)

    # 3. Add time_diff column (as timedelta64 object).
    deliveries_df["time_diff"] = deliveries_df["delivery_timestamp"] - deliveries_df["purchase_timestamp"]

    # 4. Drop rows where `time_diff` is null (which means that `delivery_timestamp` was null).
    deliveries_df = deliveries_df[deliveries_df["time_diff"].notna()]

    # 5. Change type of `time_diff` from timedelta64 to seconds in float.
    # time diff as duration in seconds
    deliveries_df["time_diff"] = deliveries_df["time_diff"].apply(datetime.timedelta.total_seconds)

    # drop rows where event_type is not equal "BUY_PRODUCT"
    sessions_df = sessions_df[sessions_df["event_type"] == "BUY_PRODUCT"]
    df = deliveries_df.merge(sessions_df, on="purchase_id", how="left")

    # making sure, that timestamp == purchase_timestamp
    num_of_rows_before = df.shape[0]
    df = df[df["timestamp"] == df["purchase_timestamp"]]
    num_of_rows_after = df.shape[0]
    assert num_of_rows_before == num_of_rows_after

    # now we can drop timestamp column, as it is redundant
    df = df.drop(columns="timestamp")

    df = df.merge(users_df, on="user_id", how="left")
    df = df.merge(products_df, on="product_id", how="left")

    # rejecting outliers for given PRICE_TRESHOLD
    df = df[df["price"] <= config.PRICE_TRESHOLD]

    # rejecting outliers for given WEIGHT_TRESHOLD
    df = df[df["weight_kg"] <= config.WEIGHT_TRESHOLD]

    # deleting rows with prices below 0
    df = df[df["price"] >= 0]

    # deleting rows with time_diff below 0
    df = df[df["time_diff"] >= 0]

    # adding column with day of week
    df["day_of_week"] = df["purchase_timestamp"].dt.dayofweek

    # adding city_and_street interaction column
    df["city_and_street"] = df["city"] + " " + df["street"]

    # adding continuous variable from purchase_timestamp (days from the first date)
    df["purchase_datetime_delta"] = (df["purchase_timestamp"] - df["purchase_timestamp"].min()) / np.timedelta64(1, "D")

    return df


@task
def drop_columns(data: pd.DataFrame, columns: list):
    """Drop unimportant columns

    Parameters
    ----------
    data : pd.DataFrame
        Data to process
    columns : list
        Columns to drop
    """
    return data.drop(columns=columns)


def _one_hot_encode_single_col(df: pd.DataFrame, col_name: str):
    """One hot encode single column (used in one hot encoding function)

    Parameters
    ----------
    df : pd.DataFrame
        Data to process
    col_name : str
        Name of the column to perform one-hot encoding on
    """
    one_hot = pd.get_dummies(df[col_name], drop_first=False)
    df = df.drop(columns=col_name)
    df = df.join(one_hot)
    df = df.drop_duplicates()
    return df


@task
def one_hot_encoding(data: pd.DataFrame, config: ProcessConfig):
    """One hot encoding

    Parameters
    ----------
    data : pd.DataFrame
        Data to process
    config : ProcessConfig
        config object with constants
    """
    cols = config.one_hot_columns
    cols.extend(config.drop_columns)
    cols = set(cols)
    cols_in_df = set(data.columns.values.tolist())
    cols_to_one_hot = cols.intersection(cols_in_df)
    for col_name in cols_to_one_hot:
        data = _one_hot_encode_single_col(data, col_name)
    data = data.dropna()
    return data


@task
def normalize_min_max(data: pd.DataFrame, config: ProcessConfig):
    """Performs minmax normalization on specified columns

    Parameters
    ----------
    data : pd.DataFrame
        Data to process
    config : ProcessConfig
        config object with constants
    """
    # specify columns for min-max scaling
    cols_to_min_max = config.min_max_columns
    cols_in_df = set(data.columns.values.tolist())
    cols_to_min_max = cols_to_min_max.intersection(cols_in_df)

    for col in cols_to_min_max:
        x = data[col].values
        min_max_scaler = MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x.reshape(-1, 1))
        data[col] = x_scaled
    return data


@task
def get_X_y(data: pd.DataFrame, label: str):
    """Get features and label

    Parameters
    ----------
    data : pd.DataFrame
        Data to process
    label : str
        Name of the label
    """
    X = data.drop(columns=label)
    y = data[label]
    return X, y


@task
def split_train_test(X: pd.DataFrame, y: pd.DataFrame, test_size: int, seed: int):
    """_summary_

    Parameters
    ----------
    X : pd.DataFrame
        Features
    y : pd.DataFrame
        Target
    test_size : int
        Size of the test set
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


@task
def save_processed_data(data: dict, save_location: str):
    """Save processed data

    Parameters
    ----------
    data : dict
        Data to process
    save_location : str
        Where to save the data
    """
    joblib.dump(data, save_location)


@flow
def process(
    location: Location = Location(),
    config: ProcessConfig = ProcessConfig(),
):
    """Flow to process the ata

    Parameters
    ----------
    location : Location, optional
        Locations of inputs and outputs, by default Location()
    config : ProcessConfig, optional
        Configurations for processing data, by default ProcessConfig()
    """
    sessions_df, deliveries_df, products_df, users_df = get_raw_data(location.data_raw)
    data = merge_data(sessions_df, deliveries_df, products_df, users_df, config)
    processed = drop_columns(data, config.drop_columns)
    processed = one_hot_encoding(processed, config)
    processed = normalize_min_max(processed, config)
    X, y = get_X_y(processed, config.label)
    split_data = split_train_test(X, y, config.test_size, config.SEED)
    save_processed_data(split_data, location.data_process)


if __name__ == "__main__":
    process()