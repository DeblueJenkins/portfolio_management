from models.data.source import Eikon
import os
import argparse
from dataclasses import dataclass
from copy import deepcopy
from typing import List
@dataclass
class Downloader:

    rics: List
    tr_field: str
    col_field: str
    start_date: str
    end_date: str
    save: bool
    overwrite: bool
    path_api: str
    path_data: str
    col_date: str = 'Calc Date'

    def __post_init__(self):

        self.api = Eikon(self.path_api)
        self.date_field = f"{self.tr_field}.calcdate"

        self.save_config = {
            'save': self.save,
            'path': fr'{self.path_data}\{self.col_field}'
        }


        self._params_download = {
            'rics': self.rics,
            'field': [self.tr_field, self.col_field],
            'date_field': [self.date_field, self.col_date],
            'save_config': self.save_config,
            'params': {
                'SDate': self.start_date,
                'EDate': self.end_date,
            }
        }

        self._params_load = deepcopy(self._params_download)
        self._params_load['load_data'] = self.save_config['path']

        del self._params_load['save_config']
        del self._params_load['params']

        self._check_path()

    def run(self):

        self.api.download_timeseries(**self._params_download)

    def load(self):

        self.api.load_timeseries(**self._params_load)

    def _check_path(self):

        if os.path.exists(self.save_config['path']):
            if self.overwrite:
                os.remove(self.save_config['path'])
                os.makedirs(self.save_config['path'])
        else:
            os.makedirs(self.save_config['path'])




    @staticmethod
    def get_constituents(index: str, api: Eikon, date: str):

        constituents = {}
        logger = []
        if isinstance(index, str):
            constituents[index] = api.get_index_constituents(index, date=date)
        elif isinstance(index, list):
            for ind in index:
                try:
                    constituents[ind] = api.get_index_constituents(ind, date=date)
                except Exception as e:
                    logger.extend(e)
                    continue

        else:
            raise Exception('index provided must be either one index (str) or list of multiple indices (List[str])')

        out = []
        for k,v in constituents.items():
            out.extend(v)

        return logger, out


def main(start_date, end_date, tr_field, col_field):

    path_data = r'C:\Users\serge\OneDrive\portfolio_management\data'
    path_api = r'C:\Users\serge\OneDrive\reuters\apikeys.csv'


    _, constituents = Downloader.get_constituents(index='.SPX', api=Eikon(path_api), date=end_date)
    print(f'Error {_}')

    params = {
        'rics': constituents,
        'tr_field': tr_field,
        'col_field': col_field,
        'start_date': end_date,
        'end_date': start_date,
        'save': True,
        'overwrite': True,
        'path_api': path_api,
        'path_data': path_data
    }

    entrypoint_download = Downloader(**params)
    entrypoint_download.run()

if __name__ == '__main__':

    # parser = argparse.ArgumentParser()
    #
    # parser.add_argument("-s", "--start_date", type=str, required=True, help="Start Date")
    # parser.add_argument("-e", "--end_date", type=str, required=True, help="End Date")
    # parser.add_argument("-tr", "--tr_field", type=str, required=True, help="TR. field")
    # parser.add_argument("-col", "--col_field", type=str, required=True, help="Name of field")
    #
    # args = parser.parse_args()

    start_date = '2000-12-31'
    end_date = '2025-01-08'
    tr_field = 'TR.GrossProfit'
    col_field = 'GrossProfit'

    args = [start_date, end_date, tr_field, col_field]

    main(*args)






