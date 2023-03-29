Module src.train_model
======================
Python script to train the model

Functions
---------

    
`get_processed_data(data_location: str)`
:   Get processed data from a specified location
    
    Parameters
    ----------
    data_location : str
        Location to get the data

    
`log_percent_of_good_predictions(model: sklearn.model_selection._search.GridSearchCV, predictions: <built-in function array>, y_test: <built-in function array>, error=86400)`
:   Show percent of good predictions for +- 24 hours
    
    Parameters
    ----------
    model : GridSearchCV
        trained model
    predictions : np.array
        numpy array with predictions
    y_test : np.array
    error : int
        +- error in seconds

    
`predict(grid: sklearn.model_selection._search.GridSearchCV, X_test: pandas.core.frame.DataFrame)`
:   _summary_
    
    Parameters
    ----------
    grid : GridSearchCV
    X_test : pd.DataFrame
        Features for testing

    
`save_model(model: sklearn.model_selection._search.GridSearchCV, save_path: str)`
:   Save model to a specified location
    
    Parameters
    ----------
    model : GridSearchCV
    save_path : str

    
`save_predictions(predictions: <built-in function array>, save_path: str)`
:   Save predictions to a specified location
    
    Parameters
    ----------
    predictions : np.array
    save_path : str

    
`train(location: config.Location = Location(data_raw={'deliveries_path': 'data/raw/deliveries.jsonl', 'products_path': 'data/raw/products.jsonl', 'sessions_path': 'data/raw/sessions.jsonl', 'users_path': 'data/raw/users.jsonl'}, data_process='data/processed/xy.pkl', data_final='data/final/predictions.pkl', model='models/model.pkl', encoder='models/encoder.pkl', scaler='models/scaler.pkl', preprocessor='models/preprocessor.pkl', min_purchase_timestamp='models/min_purchase_timestamp.pkl', input_notebook='notebooks/data_analysis.ipynb', output_notebook='notebooks/data_analysis_results.ipynb'), model_params: config.ModelParams = ModelParams(NUM_OF_HOURS=24, alpha=[0.01, 0.1]))`
:   Flow to train the model
    
    Parameters
    ----------
    location : Location, optional
        Locations of inputs and outputs, by default Location()
    model_params : ModelParams, optional
        Configurations for training the model, by default ModelParams()

    
`train_model(model_params: config.ModelParams, X_train: pandas.core.frame.DataFrame, y_train: pandas.core.series.Series)`
:   Train the model
    
    Parameters
    ----------
    model_params : ModelParams
        Parameters for the model
    X_train : pd.DataFrame
        Features for training
    y_train : pd.Series
        Label for training