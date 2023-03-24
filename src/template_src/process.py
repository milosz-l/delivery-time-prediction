"""Python script to process the data"""

import joblib
import pandas as pd
from config import Location, ProcessConfig
from prefect import flow, task
from sklearn.model_selection import train_test_split
from typing import List
import numpy as np


@task
def get_raw_data(data_locations: List[str]):
    """Read raw data

    Parameters
    ----------
    data_locations : List[str]
        List with the locations of the raw data
    """
    # fact table
    sessions_df = pd.read_json("data/sessions.jsonl", lines=True)

    # dimension tables
    deliveries_df = pd.read_json("data/deliveries.jsonl", lines=True)
    products_df = pd.read_json("data/products.jsonl", lines=True)
    users_df = pd.read_json("data/users.jsonl", lines=True)
    return sessions_df, deliveries_df, products_df, users_df

@task
def merge_data(sessions_df, deliveries_df, products_df, users_df):
    """Merge fact table and dimension tables

    Parameters
    ----------
    session_df : DataFrame
        fact table
    deliveries_df : DataFrame
        dimension table
    products_df : DataFrame
        dimension table
    users_df : DataFrame
        dimension table
    """
    # 1. Cut microseconds from `delivery_timestamp`, so it will be the same format as `purchase_timestamp`, because there are no microseconds in purchase_timestamp (using "." as a separator).
    deliveries_df["delivery_timestamp"] = deliveries_df["delivery_timestamp"].str.split('.', expand=True)[0]

    # 2. Change columns format to datetime.
    deliveries_df["purchase_timestamp"] = pd.to_datetime(deliveries_df["purchase_timestamp"], format=DATE_FORMAT)
    deliveries_df["delivery_timestamp"] = pd.to_datetime(deliveries_df["delivery_timestamp"], format=DATE_FORMAT)

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
    assert(num_of_rows_before == num_of_rows_after)

    # now we can drop timestamp column, as it is redundant
    df = df.drop(columns="timestamp")

    df = df.merge(users_df, on="user_id", how="left")
    df = df.merge(products_df, on="product_id", how="left")

    # rejecting outliers for given PRICE_TRESHOLD
    df = df[df["price"] <= PRICE_TRESHOLD]

    # rejecting outliers for given WEIGHT_TRESHOLD
    df = df[df["weight_kg"] <= WEIGHT_TRESHOLD]

    # deleting rows with prices below 0
    df = df[df["price"] >= 0]

    # deleting rows with time_diff below 0
    df = df[df["time_diff"] >= 0]

    # adding column with day of week
    df['day_of_week'] = df['purchase_timestamp'].dt.dayofweek

    # adding city_and_street interaction column
    df['city_and_street'] = df['city'] + ' ' + df['street']

    # adding continuous variable from purchase_timestamp (days from the first date)
    df['purchase_datetime_delta'] = (df['purchase_timestamp'] - df['purchase_timestamp'].min()) / np.timedelta64(1, 'D')

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
def split_train_test(X: pd.DataFrame, y: pd.DataFrame, test_size: int):
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
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=0
    )
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
    data = merge_data(sessions_df, deliveries_df, products_df, users_df)
    processed = drop_columns(data, config.drop_columns)
    X, y = get_X_y(processed, config.label)
    split_data = split_train_test(X, y, config.test_size)
    save_processed_data(split_data, location.data_process)


if __name__ == "__main__":
    process(config=ProcessConfig(test_size=0.1))
