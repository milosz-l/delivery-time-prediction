from fileinput import filename
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as ss
import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from src.TimeDiffConstants import DATE_FORMAT, PRICE_TRESHOLD, WEIGHT_TRESHOLD, NUM_OF_HOURS, SEED, COLS_TO_DROP_ALWAYS


class TimeDiffDataTransformer:
    def __init__(self, request_df=None):
        # fact table
        sessions_df = pd.read_json("data/sessions.jsonl", lines=True)

        # dimension tables
        deliveries_df = pd.read_json("data/deliveries.jsonl", lines=True)
        products_df = pd.read_json("data/products.jsonl", lines=True)
        users_df = pd.read_json("data/users.jsonl", lines=True)

        self.df = TimeDiffDataTransformer._prepare_df(sessions_df, deliveries_df, products_df, users_df)

        if type(request_df) == pd.DataFrame:
            self.df = pd.concat([self.df, request_df], ignore_index=True)

    def _prepare_df(sessions_df, deliveries_df, products_df, users_df):
        # 1.
        deliveries_df["delivery_timestamp"] = deliveries_df["delivery_timestamp"].str.split('.', expand=True)[0]

        # 2.
        deliveries_df["purchase_timestamp"] = pd.to_datetime(deliveries_df["purchase_timestamp"], format=DATE_FORMAT)
        deliveries_df["delivery_timestamp"] = pd.to_datetime(deliveries_df["delivery_timestamp"], format=DATE_FORMAT)

        # 3.
        deliveries_df["time_diff"] = deliveries_df["delivery_timestamp"] - deliveries_df["purchase_timestamp"]

        # 4.
        deliveries_df = deliveries_df[deliveries_df["time_diff"].notna()]

        # 5.
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

    def drop_cols(self, additional_cols=None):
        if additional_cols is None:
            additional_cols = ["city_and_street",
                               'product_name',
                               'product_id',
                               'brand',
                               'category_path']
        cols_to_drop = list(COLS_TO_DROP_ALWAYS)
        cols_to_drop.extend(additional_cols)
        self.df = self.df.drop(columns=cols_to_drop)

    def _one_hot_encoding_single_col(df, col_name):
        one_hot = pd.get_dummies(df[col_name], drop_first=False)
        df = df.drop(columns=col_name)
        df = df.join(one_hot)
        df = df.drop_duplicates()
        return df

    def one_hot_encoding_columns(self):
        cols = ["delivery_company",
                "city",
                "street",
                "city_and_street",
                'brand',
                'product_name',
                "category_path",
                'day_of_week',
                'product_id']
        cols.extend(COLS_TO_DROP_ALWAYS)
        cols = set(cols)
        cols_in_df = set(self.df.columns.values.tolist())
        cols_to_one_hot = cols.intersection(cols_in_df)
        # print(self.df.columns)
        for col_name in cols_to_one_hot:
            self.df = TimeDiffDataTransformer._one_hot_encoding_single_col(self.df, col_name)
        self.df = self.df.dropna()

    def normalize_min_max(self):
        # specify columns for min-max scaling
        cols_to_min_max = set(['price', 'weight_kg', 'purchase_datetime_delta', 'offered_discount'])
        cols_in_df = set(self.df.columns.values.tolist())
        cols_to_min_max = cols_to_min_max.intersection(cols_in_df)

        for col in cols_to_min_max:
            x = self.df[col].values
            min_max_scaler = MinMaxScaler()
            x_scaled = min_max_scaler.fit_transform(x.reshape(-1, 1))
            self.df[col] = x_scaled

    def make_all_transformations(self, additional_cols_to_drop=None):
        self.drop_cols(additional_cols=additional_cols_to_drop)
        self.one_hot_encoding_columns()
        self.normalize_min_max()

    def get_df(self):
        return self.df.copy()

    def to_csv(self, file_name='df_from_TimeDiffDataTransformer.csv'):
        self.df.to_csv(filename)
