import logging

import joblib
import pandas as pd
from flask import Flask, jsonify, request

from config import AppConfig, Location
from process import process_query

app = Flask(__name__)
location_config = Location()
app_config = AppConfig()


@app.route("/", methods=["POST"])
def predict():
    # prepare data
    json_ = request.json
    # query_df = pd.DataFrame(json_)
    query_df = process_query(json_)

    # load model
    model = joblib.load(location_config.model)
    prediction = model.predict(query_df)
    logging.info("prediction:\n", prediction, "\ngenerated for:\n", query_df)
    return jsonify({"prediction": list(prediction)})


if __name__ == "__main__":
    app.run(port=app_config.port, debug=True)
