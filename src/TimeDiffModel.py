from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd


class TimeDiffModel:

    def __init__(self, df):
        self.df = df
