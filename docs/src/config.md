Module src.config
=================
create Pydantic models

Functions
---------

    
`must_be_non_negative(v: float) ‑> float`
:   Check if the v is non-negative
    
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

Classes
-------

`AppConfig(**data: Any)`
:   Specify the parameters of web service for model deployment
    
    Create a new model by parsing and validating input data from keyword arguments.
    
    Raises ValidationError if the input data cannot be parsed to form a valid model.

    ### Ancestors (in MRO)

    * pydantic.main.BaseModel
    * pydantic.utils.Representation

`Location(**data: Any)`
:   Specify the locations of inputs and outputs
    
    Create a new model by parsing and validating input data from keyword arguments.
    
    Raises ValidationError if the input data cannot be parsed to form a valid model.

    ### Ancestors (in MRO)

    * pydantic.main.BaseModel
    * pydantic.utils.Representation

    ### Class variables

    `data_final: str`
    :

    `data_process: str`
    :

    `data_raw: dict`
    :

    `encoder: str`
    :

    `input_notebook: str`
    :

    `min_purchase_timestamp: str`
    :

    `model: str`
    :

    `output_notebook: str`
    :

    `preprocessor: str`
    :

    `scaler: str`
    :

`ModelParams(**data: Any)`
:   Specify the parameters of the `train` flow
    
    Create a new model by parsing and validating input data from keyword arguments.
    
    Raises ValidationError if the input data cannot be parsed to form a valid model.

    ### Ancestors (in MRO)

    * pydantic.main.BaseModel
    * pydantic.utils.Representation

    ### Class variables

    `NUM_OF_HOURS: int`
    :

    `alpha: List[float]`
    :

`ProcessConfig(**data: Any)`
:   Specify the parameters of the `process` flow
    
    Create a new model by parsing and validating input data from keyword arguments.
    
    Raises ValidationError if the input data cannot be parsed to form a valid model.

    ### Ancestors (in MRO)

    * pydantic.main.BaseModel
    * pydantic.utils.Representation

    ### Class variables

    `DATE_FORMAT: str`
    :

    `PRICE_TRESHOLD: int`
    :

    `SEED: int`
    :

    `TEST_SIZE: float`
    :

    `WEIGHT_TRESHOLD: int`
    :

    `drop_columns: List[str]`
    :

    `label: str`
    :

    `min_max_columns: List[str]`
    :

    `one_hot_columns: List[str]`
    :

    `test_size: float`
    :