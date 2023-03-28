"""
create Pydantic models
"""
from random import randint
from typing import List

from pydantic import BaseModel, validator


def must_be_non_negative(v: float) -> float:
    """Check if the v is non-negative

    Parameters
    ----------
    v : float
        value

    Returns
    -------
    float
        v

    Raises
    ------
    ValueError
        Raises error when v is negative
    """
    if v < 0:
        raise ValueError(f"{v} must be non-negative")
    return v


class Location(BaseModel):
    """Specify the locations of inputs and outputs"""

    data_raw: dict = {
        "deliveries_path": "data/raw/deliveries.jsonl",
        "products_path": "data/raw/products.jsonl",
        "sessions_path": "data/raw/sessions.jsonl",
        "users_path": "data/raw/users.jsonl",
    }
    data_process: str = "data/processed/xy.pkl"
    data_final: str = "data/final/predictions.pkl"
    model: str = "models/model.pkl"
    encoder: str = "models/encoder.pkl"
    scaler: str = "models/scaler.pkl"
    preprocessor: str = "models/preprocessor.pkl"
    min_purchase_timestamp: str = "models/min_purchase_timestamp.pkl"
    input_notebook: str = "notebooks/data_analysis.ipynb"
    output_notebook: str = "notebooks/data_analysis_results.ipynb"


class ProcessConfig(BaseModel):
    """Specify the parameters of the `process` flow"""

    DATE_FORMAT: str = "%Y-%m-%dT%H:%M:%S"
    PRICE_TRESHOLD: int = 100_000  # for outliers
    WEIGHT_TRESHOLD: int = 50  # for outliers
    SEED: int = 23  # randint(0, 100000)
    TEST_SIZE: float = 0.001

    drop_columns: List[str] = [
        "delivery_timestamp",
        "session_id",
        "purchase_id",
        "event_type",
        "name",
        "user_id",
        "offered_discount",
        "optional_attributes",
        "purchase_timestamp",
    ]
    one_hot_columns: List[str] = [
        "delivery_company",
        "city",
        "street",
        "city_and_street",
        "brand",
        "product_name",
        "category_path",
        "day_of_week",
        "product_id",
    ]
    min_max_columns: List[str] = ["price", "weight_kg", "purchase_datetime_delta", "offered_discount"]
    label: str = "time_diff"
    test_size: float = 0.2

    _validated_test_size = validator("test_size", allow_reuse=True)(must_be_non_negative)


class ModelParams(BaseModel):
    """Specify the parameters of the `train` flow"""

    NUM_OF_HOURS: int = 24
    alpha: List[float] = [0.01, 0.1]
    # gamma: List[float] = [1, 0.1, 0.01, 0.001, 0.0001]

    _validated_fields = validator("*", allow_reuse=True, each_item=True)(must_be_non_negative)


class AppConfig(BaseModel):
    """Specify the parameters of web service for model deployment"""

    port = 5000
