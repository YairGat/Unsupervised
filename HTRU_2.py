import numpy as np

from Data import Data


class HTRU_2(Data):
    classification = []

    def __init__(self):
        super().__init__('csv/HTRU_2.csv', ',')

    def _load_csv(self):
        self.classification = self.csv[:, 8]
        self.csv = self.csv[:, [0,1,2,3,4,5,6,7]]


