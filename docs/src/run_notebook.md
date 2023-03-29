Module src.run_notebook
=======================
Python script to run the notebook

Functions
---------

    
`run_notebook(location: config.Location = Location(data_raw={'deliveries_path': 'data/raw/deliveries.jsonl', 'products_path': 'data/raw/products.jsonl', 'sessions_path': 'data/raw/sessions.jsonl', 'users_path': 'data/raw/users.jsonl'}, data_process='data/processed/xy.pkl', data_final='data/final/predictions.pkl', model='models/model.pkl', encoder='models/encoder.pkl', scaler='models/scaler.pkl', preprocessor='models/preprocessor.pkl', min_purchase_timestamp='models/min_purchase_timestamp.pkl', input_notebook='notebooks/data_analysis.ipynb', output_notebook='notebooks/data_analysis_results.ipynb'))`
:   Run a notebook with specified parameters then
    generate a notebook with the outputs
    
    Parameters
    ----------
    location : Location, optional
        Locations of inputs and outputs, by default Location()