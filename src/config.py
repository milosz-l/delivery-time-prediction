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

    data_raw: List[str] = [
        "data/raw/deliveries.jsonl",
        "data/raw/products.jsonl",
        "data/raw/sessions.jsonl",
        "data/raw/users.jsonl",
    ]
    data_process: str = "data/processed/xy.pkl"
    data_final: str = "data/final/predictions.pkl"
    model: str = "models/model.pkl"
    input_notebook: str = "notebooks/analyze_results.ipynb"
    output_notebook: str = "notebooks/results.ipynb"


class ProcessConfig(BaseModel):
    """Specify the parameters of the `process` flow"""

    DATE_FORMAT: str = "%Y-%m-%dT%H:%M:%S"
    PRICE_TRESHOLD: int = 100_000  # for outliers
    WEIGHT_TRESHOLD: int = 50  # for outliers
    NUM_OF_HOURS: int = 24
    SEED: int = randint(0, 100000)  # 42
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
    label: str = "Species"
    test_size: float = 0.3

    _validated_test_size = validator("test_size", allow_reuse=True)(must_be_non_negative)


class ModelParams(BaseModel):
    """Specify the parameters of the `train` flow"""

    C: List[float] = [0.1, 1, 10, 100, 1000]
    gamma: List[float] = [1, 0.1, 0.01, 0.001, 0.0001]

    _validated_fields = validator("*", allow_reuse=True, each_item=True)(must_be_non_negative)
