Module src.process
==================
Python script to process the data

Functions
---------

    
`create_preprocessor(X: pandas.core.frame.DataFrame, config: config.ProcessConfig)`
:   Creates sklearn preprocessor
    
    Parameters
    ----------
    X : pd.DataFrame
        dataframe with training data
    config : ProcessConfig, optional
        Configurations for processing data, by default ProcessConfig()

    
`drop_columns(data: pandas.core.frame.DataFrame, columns: list)`
:   Drop unimportant columns
    
    Parameters
    ----------
    data : pd.DataFrame
        Data to process
    columns : list
        Columns to drop

    
`feature_engineering(df: pandas.core.frame.DataFrame)`
:   Perform feature engineering on given DataFrame
    
    Parameters
    ----------
    df : pd.DataFrame

    
`get_X_y(data: pandas.core.frame.DataFrame, label: str)`
:   Get features and label
    
    Parameters
    ----------
    data : pd.DataFrame
        Data to process
    label : str
        Name of the label

    
`get_raw_data(data_locations: dict)`
:   Read raw data
    
    Parameters
    ----------
    data_locations : dict
        Dictionary with the locations of the raw data

    
`prepare_test_df(query: pandas.core.frame.DataFrame, config: config.ProcessConfig, location: config.Location)`
:   Prepare query for preprocessing
    
    Parameters
    ----------
    query : pd.DataFrame
        Query for prediction (single line as a DataFrame)
    config : ProcessConfig
        config object with constants
    location : Location
        Locations of inputs and outputs, by default Location()

    
`prepare_training_df(sessions_df: pandas.core.frame.DataFrame, deliveries_df: pandas.core.frame.DataFrame, products_df: pandas.core.frame.DataFrame, users_df: pandas.core.frame.DataFrame, config: config.ProcessConfig, location: config.Location)`
:   Merge fact table and dimension tables, then prepare training DataFrame
    
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

    
`process(location: config.Location = Location(data_raw={'deliveries_path': 'data/raw/deliveries.jsonl', 'products_path': 'data/raw/products.jsonl', 'sessions_path': 'data/raw/sessions.jsonl', 'users_path': 'data/raw/users.jsonl'}, data_process='data/processed/xy.pkl', data_final='data/final/predictions.pkl', model='models/model.pkl', encoder='models/encoder.pkl', scaler='models/scaler.pkl', preprocessor='models/preprocessor.pkl', min_purchase_timestamp='models/min_purchase_timestamp.pkl', input_notebook='notebooks/data_analysis.ipynb', output_notebook='notebooks/data_analysis_results.ipynb'), config: config.ProcessConfig = ProcessConfig(DATE_FORMAT='%Y-%m-%dT%H:%M:%S', PRICE_TRESHOLD=100000, WEIGHT_TRESHOLD=50, SEED=23, TEST_SIZE=0.001, drop_columns=['delivery_timestamp', 'session_id', 'purchase_id', 'event_type', 'name', 'user_id', 'offered_discount', 'optional_attributes', 'purchase_timestamp'], one_hot_columns=['delivery_company', 'city', 'street', 'city_and_street', 'brand', 'product_name', 'category_path', 'day_of_week', 'product_id'], min_max_columns=['price', 'weight_kg', 'purchase_datetime_delta', 'offered_discount'], label='time_diff', test_size=0.2))`
:   Flow to process the data
    
    Parameters
    ----------
    location : Location, optional
        Locations of inputs and outputs, by default Location()
    config : ProcessConfig, optional
        Configurations for processing data, by default ProcessConfig()

    
`process_query(query: List, config: config.ProcessConfig = ProcessConfig(DATE_FORMAT='%Y-%m-%dT%H:%M:%S', PRICE_TRESHOLD=100000, WEIGHT_TRESHOLD=50, SEED=23, TEST_SIZE=0.001, drop_columns=['delivery_timestamp', 'session_id', 'purchase_id', 'event_type', 'name', 'user_id', 'offered_discount', 'optional_attributes', 'purchase_timestamp'], one_hot_columns=['delivery_company', 'city', 'street', 'city_and_street', 'brand', 'product_name', 'category_path', 'day_of_week', 'product_id'], min_max_columns=['price', 'weight_kg', 'purchase_datetime_delta', 'offered_discount'], label='time_diff', test_size=0.2), location_config: config.Location = Location(data_raw={'deliveries_path': 'data/raw/deliveries.jsonl', 'products_path': 'data/raw/products.jsonl', 'sessions_path': 'data/raw/sessions.jsonl', 'users_path': 'data/raw/users.jsonl'}, data_process='data/processed/xy.pkl', data_final='data/final/predictions.pkl', model='models/model.pkl', encoder='models/encoder.pkl', scaler='models/scaler.pkl', preprocessor='models/preprocessor.pkl', min_purchase_timestamp='models/min_purchase_timestamp.pkl', input_notebook='notebooks/data_analysis.ipynb', output_notebook='notebooks/data_analysis_results.ipynb'))`
:   Process data for prediction
    
    Parameters
    ----------
    query : List
        List with data given in the post request
    config : ProcessConfig, optional
        Configurations for processing data, by default ProcessConfig()
    location_config : Location, optional
        Locations of inputs and outputs, by default Location()

    
`save_processed_data(data: dict, save_location: str)`
:   Save processed data
    
    Parameters
    ----------
    data : dict
        Data to process
    save_location : str
        Where to save the data

    
`specify_cols_for_one_hot(data: pandas.core.frame.DataFrame, config: config.ProcessConfig)`
:   Specify columns for one-hot encoding
    
    Parameters
    ----------
    data : pd.DataFrame
        Data to process
    config : ProcessConfig
        config object with constants

    
`specify_columns_for_scaling(min_max_columns: List[str], data: pandas.core.frame.DataFrame)`
:   Determines which columns to scale based on intersection of min_max columns given in config and columns in df
    
    Parameters
    ----------
    min_max_columns : List[str]
        columns for min_max scaling specified in config
    data : pd.DataFrame
        Data to process

    
`split_train_test(X: pandas.core.frame.DataFrame, y: pandas.core.frame.DataFrame, test_size: int, seed: int)`
:   _summary_
    
    Parameters
    ----------
    X : pd.DataFrame
        Features
    y : pd.DataFrame
        Target
    test_size : int
        Size of the test set