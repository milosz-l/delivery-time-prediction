"""Python script to process the data"""

import datetime
from typing import List

import joblib
import numpy as np
import pandas as pd
from prefect import flow, get_run_logger, task
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

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


def feature_engineering(df: pd.DataFrame):
    """Perform feature engineering on given DataFrame

    Parameters
    ----------
    df : pd.DataFrame
    """
    # adding column with day of week
    df["day_of_week"] = df["purchase_timestamp"].dt.dayofweek

    # adding city_and_street interaction column
    df["city_and_street"] = df["city"] + " " + df["street"]

    return df


@task
def prepare_training_df(
    sessions_df: pd.DataFrame,
    deliveries_df: pd.DataFrame,
    products_df: pd.DataFrame,
    users_df: pd.DataFrame,
    config: ProcessConfig,
    location: Location,
):
    """Merge fact table and dimension tables, then prepare training DataFrame

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
    location : Location
        Locations of inputs and outputs, by default Location()
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

    df = feature_engineering(df)

    # adding continuous variable from purchase_timestamp (days from the first date)
    min_purchase_timestamp = df["purchase_timestamp"].min()
    df["purchase_datetime_delta"] = (df["purchase_timestamp"] - min_purchase_timestamp) / np.timedelta64(1, "D")
    joblib.dump(min_purchase_timestamp, location.min_purchase_timestamp)

    return df


@task
def prepare_test_df(query: pd.DataFrame, config: ProcessConfig, location: Location):
    """Prepare query for preprocessing

    Parameters
    ----------
    query : pd.DataFrame
        Query for prediction (single line as a DataFrame)
    config : ProcessConfig
        config object with constants
    location : Location
        Locations of inputs and outputs, by default Location()
    """
    # Change columns format to datetime.
    query["purchase_timestamp"] = pd.to_datetime(query["purchase_timestamp"], format=config.DATE_FORMAT)

    # drop rows where event_type is not equal "BUY_PRODUCT"
    query = query[query["event_type"] == "BUY_PRODUCT"]

    query = feature_engineering(query)

    min_purchase_timestamp = joblib.load(location.min_purchase_timestamp)
    query["purchase_datetime_delta"] = (query["purchase_timestamp"] - min_purchase_timestamp) / np.timedelta64(1, "D")

    return query


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


def specify_cols_for_one_hot(data: pd.DataFrame, config: ProcessConfig):
    """Specify columns for one-hot encoding

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
    cols_to_one_hot = list(cols_to_one_hot)
    return cols_to_one_hot


def specify_columns_for_scaling(min_max_columns: List[str], data: pd.DataFrame):
    """Determines which columns to scale based on intersection of min_max columns given in config and columns in df

    Parameters
    ----------
    min_max_columns : List[str]
        columns for min_max scaling specified in config
    data : pd.DataFrame
        Data to process
    """
    cols_to_min_max = set(min_max_columns)
    cols_in_df = set(data.columns.values.tolist())
    cols_to_min_max = cols_to_min_max.intersection(cols_in_df)
    return list(cols_to_min_max)


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


@task
def create_preprocessor(X: pd.DataFrame, config: ProcessConfig):
    """Creates sklearn preprocessor

    Parameters
    ----------
    X : pd.DataFrame
        dataframe with training data
    config : ProcessConfig, optional
        Configurations for processing data, by default ProcessConfig()
    """
    categorical_cols = specify_cols_for_one_hot(X, config)
    numerical_cols = specify_columns_for_scaling(config.min_max_columns, X)
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_cols,
            ),  # categorical_cols is a list of categorical columns
            ("num", MinMaxScaler(), numerical_cols),  # numerical_cols is a list of numerical columns
        ],
        # remainder="passthrough"
    )
    return preprocessor


@flow
def process(location: Location = Location(), config: ProcessConfig = ProcessConfig()):
    """Flow to process the data

    Parameters
    ----------
    location : Location, optional
        Locations of inputs and outputs, by default Location()
    config : ProcessConfig, optional
        Configurations for processing data, by default ProcessConfig()
    """
    sessions_df, deliveries_df, products_df, users_df = get_raw_data(location.data_raw)
    data = prepare_training_df(sessions_df, deliveries_df, products_df, users_df, config, location)
    data = drop_columns(data, config.drop_columns)
    X, y = get_X_y(data, config.label)

    preprocessor = create_preprocessor(X, config)

    # Fit the preprocessor to your training data and transform the data
    X_train_transformed = preprocessor.fit_transform(X)
    logger = get_run_logger()
    logger.info(f"preprocessor's transformers:\n{preprocessor.transformers_}")
    joblib.dump(preprocessor, location.preprocessor)

    split_data = split_train_test(X_train_transformed, y, config.test_size, config.SEED)
    save_processed_data(split_data, location.data_process)


@flow
def process_query(query: List, config: ProcessConfig = ProcessConfig(), location_config: Location = Location()):
    """Process data for prediction

    Parameters
    ----------
    query : List
        List with data given in the post request
    config : ProcessConfig, optional
        Configurations for processing data, by default ProcessConfig()
    location_config : Location, optional
        Locations of inputs and outputs, by default Location()
    """
    logger = get_run_logger()
    query_df = pd.DataFrame(query)
    query_df = prepare_test_df(query_df, config, location_config)
    logger.info(f"query before preprocessing:\n{query_df}")
    preprocessor = joblib.load(location_config.preprocessor)
    query_df = preprocessor.transform(query_df)
    logger.info(f"columns in encoder: {preprocessor}")
    logger.info(f"query after preprocessing:\n{query_df}")
    return query_df


if __name__ == "__main__":
    process()
