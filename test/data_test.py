"""
Project:    stock_prediction
File:       data_test.py
Created by: louise
On:         05/03/18
At:         2:14 PM
"""

import unittest
from torch.utils.data import DataLoader

from src.data import SP500Multistep


class TestSP500Multistep(unittest.TestCase):
    def test_SP500Multistep(self):
        # Parameters
        batch_size = 16
        symbols = ['GOOGL', 'AAPL', 'AMZN', 'FB', 'ZION', 'NVDA', 'GS']
        T = 10
        start_date = '2013-01-01'
        end_date = '2013-12-31'
        n_step_data = 1

        # training data
        dset = SP500Multistep('data/sandp500/individual_stocks_5yr',
                              symbols=symbols,
                              start_date=start_date,
                              end_date=end_date,
                              T=T,
                              step=n_step_data,
                              n_in=T)
        train_loader = DataLoader(dset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=4,
                                  pin_memory=True  # CUDA only
                                  )
        x1, y1 = train_loader.dataset[0]
        x2, y2 = train_loader.dataset[1]
        self.assertEqual(x1[:, -1].numpy().all(), x2[:, -2].numpy().all())
        self.assertEqual(y1[:, -1].numpy().all(), y2[:, -2].numpy().all())


if __name__ == '__main__':
    unittest.main()
