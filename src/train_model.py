"""Python script to train the model"""
import joblib
import numpy as np
import pandas as pd
from prefect import flow, get_run_logger, task
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

from config import Location, ModelParams


@task
def get_processed_data(data_location: str):
    """Get processed data from a specified location

    Parameters
    ----------
    data_location : str
        Location to get the data
    """
    return joblib.load(data_location)


@task
def train_model(model_params: ModelParams, X_train: pd.DataFrame, y_train: pd.Series):
    """Train the model

    Parameters
    ----------
    model_params : ModelParams
        Parameters for the model
    X_train : pd.DataFrame
        Features for training
    y_train : pd.Series
        Label for training
    """
    # grid = GridSearchCV(Ridge(), model_params.dict(), refit=True, verbose=3)
    grid = RandomForestRegressor()
    grid.fit(X_train, y_train)
    return grid


@task
def predict(grid: GridSearchCV, X_test: pd.DataFrame):
    """_summary_

    Parameters
    ----------
    grid : GridSearchCV
    X_test : pd.DataFrame
        Features for testing
    """
    return grid.predict(X_test)


@task
def save_model(model: GridSearchCV, save_path: str):
    """Save model to a specified location

    Parameters
    ----------
    model : GridSearchCV
    save_path : str
    """
    joblib.dump(model, save_path)


@task
def save_predictions(predictions: np.array, save_path: str):
    """Save predictions to a specified location

    Parameters
    ----------
    predictions : np.array
    save_path : str
    """
    joblib.dump(predictions, save_path)


@task
def log_percent_of_good_predictions(model: GridSearchCV, predictions: np.array, y_test: np.array, error=24 * 60 * 60):
    """Show percent of good predictions for +- 24 hours

    Parameters
    ----------
    model : GridSearchCV
        trained model
    predictions : np.array
        numpy array with predictions
    y_test : np.array
    error : int
        +- error in seconds
    """
    predictions_time_diff = np.abs(y_test - predictions)
    num_of_good_predictions = (predictions_time_diff < error).sum()
    percent_of_good_predictions = num_of_good_predictions / len(predictions_time_diff)
    logger = get_run_logger()
    logger.info(
        f"number of good predictions for {type(model).__name__} = {num_of_good_predictions}/{len(predictions_time_diff)}"
    )
    logger.info(f"which is {percent_of_good_predictions * 100}% for +-{round(error/60/60)} hours\n")


@flow
def train(
    location: Location = Location(),
    model_params: ModelParams = ModelParams(),
):
    """Flow to train the model

    Parameters
    ----------
    location : Location, optional
        Locations of inputs and outputs, by default Location()
    model_params : ModelParams, optional
        Configurations for training the model, by default ModelParams()
    """
    data = get_processed_data(location.data_process)
    model = train_model(model_params, data["X_train"], data["y_train"])
    predictions = predict(model, data["X_test"])
    save_model(model, save_path=location.model)
    save_predictions(predictions, save_path=location.data_final)
    log_percent_of_good_predictions(model, predictions, data["y_test"], model_params.NUM_OF_HOURS * 60 * 60)


if __name__ == "__main__":
    train()
