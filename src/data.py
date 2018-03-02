"""
Project:    gresearch
File:       data.py
Created by: louise
On:         25/01/18
At:         4:56 PM
"""
import os

import torch
from torch.utils.data.dataset import Dataset
from sklearn.preprocessing import MinMaxScaler

import pandas as pd


class SP500(Dataset):
    def __init__(self, folder_dataset, T=10, symbols=['AAPL'], use_columns=['Date', 'Close'], start_date='2012-01-01',
                 end_date='2015-12-31', step=1):
        """

        :param folder_dataset: str
        :param T: int
        :param symbols: list of str
        :param use_columns: bool
        :param start_date: str, date format YYY-MM-DD
        :param end_date: str, date format YYY-MM-DD
        """
        self.scaler = MinMaxScaler()
        self.symbols = symbols
        if len(symbols)==0:
            print("No Symbol was specified")
            return
        self.start_date = start_date
        if len(start_date)==0:
            print("No start date was specified")
            return
        self.end_date = end_date
        if len(end_date)==0:
            print("No end date was specified")
            return
        self.use_columns = use_columns
        if len(use_columns)==0:
            print("No column was specified")
            return
        self.T = T

        # Create output dataframe
        self.dates = pd.date_range(self.start_date, self.end_date)
        self.df_data = pd.DataFrame(index=self.dates)

        # Read csv files corresponding to symbols
        for symbol in symbols:
            fn = os.path.join(folder_dataset, symbol + "_data.csv")
            fn = "/home/louise/src/gresearch/" + folder_dataset + "/" + symbol + "_data.csv"
            print(fn)
            df_current = pd.read_csv(fn, index_col='Date', usecols=self.use_columns, na_values='nan', parse_dates=True)
            df_current = df_current.rename(columns={'Close': symbol})
            self.df_data = self.df_data.join(df_current)

        # Replace NaN values with forward then backward filling
        self.df_data.fillna(method='ffill', inplace=True, axis=0)
        self.df_data.fillna(method='bfill', inplace=True, axis=0)
        self.numpy_data = self.df_data.as_matrix(columns=self.symbols)
        self.train_data = self.scaler.fit_transform(self.numpy_data)

        self.chunks = torch.FloatTensor(self.train_data).unfold(0, self.T, step).permute(0, 2, 1)

    def __getitem__(self, index):

        x = self.chunks[index, :-1, :]
        y = self.chunks[index, -1, :]
        return x, y

    def __len__(self):
        return self.chunks.size(0)


class SP500Multistep(Dataset):
    def __init__(self, folder_dataset, T=10, symbols=['AAPL'], use_columns=['Date', 'Close'], start_date='2012-01-01',
                 end_date='2015-12-31', step=1, n_in=5, n_out=5):
        """

        :param folder_dataset: str
        :param T: int
        :param symbols: list of str
        :param use_columns: bool
        :param start_date: str, date format YYY-MM-DD
        :param end_date: str, date format YYY-MM-DD
        """
        self.scaler = MinMaxScaler()
        self.symbols = symbols
        if len(symbols)==0:
            print("No Symbol was specified")
            return
        self.start_date = start_date
        if len(start_date)==0:
            print("No start date was specified")
            return
        self.end_date = end_date
        if len(end_date)==0:
            print("No end date was specified")
            return
        self.use_columns = use_columns
        if len(use_columns)==0:
            print("No column was specified")
            return
        self.T = T

        # Create output dataframe
        self.dates = pd.date_range(self.start_date, self.end_date)
        self.df_data = pd.DataFrame(index=self.dates)

        # Read csv files corresponding to symbols
        for symbol in symbols:
            fn = os.path.join(folder_dataset, symbol + "_data.csv")
            fn = "/home/louise/src/gresearch/" + folder_dataset + "/" + symbol + "_data.csv"
            print(fn)
            df_current = pd.read_csv(fn, index_col='Date', usecols=self.use_columns, na_values='nan', parse_dates=True)
            df_current = df_current.rename(columns={'Close': symbol})
            self.df_data = self.df_data.join(df_current)

        # Replace NaN values with forward then backward filling
        self.df_data.fillna(method='ffill', inplace=True, axis=0)
        self.df_data.fillna(method='bfill', inplace=True, axis=0)
        self.numpy_data = self.df_data.as_matrix(columns=self.symbols)
        self.train_data = self.scaler.fit_transform(self.numpy_data)

        self.chunks_data = torch.FloatTensor(self.train_data).unfold(0, n_in, step).permute(0, 2, 1)
        self.chunks_target = torch.FloatTensor(self.train_data).unfold(n_in, n_out, step).permute(0, 2, 1)

    def __getitem__(self, index):

        x = self.chunks[index, :-1, :]
        y = self.chunks[index, -1, :]
        return x, y

    def __len__(self):
        return self.chunks.size(0)

