import numpy as np


from Data import Data


class AllUsersData(Data):
    classification = []

    def __init__(self):
        super().__init__('csv/allUsers.lcl.csv', ',')

    def _load_csv(self):
        # column_with_missing_info = self.find_column_with_missing_information()
        column_with_missing_info = range(11, 38)
        for i in column_with_missing_info:
            index_of_missing_info = []
            index_of_info = []
            for j in range(2, self.csv.shape[0]):
                if self.csv[j][i] == '?':
                    index_of_missing_info.append(j)
                else:
                    index_of_info.append(float(self.csv[j][i]))
            median = np.median(index_of_info)
            for j in index_of_missing_info:
                self.csv[j][i] = median
        self.classification = self.csv[:, 0]
        self.csv = np.delete(self.csv, 0, 1)


    def find_column_with_missing_information(self):
        column_with_missing_info = []
        for i in range(self.csv.shape[1]):
            for j in range(self.csv.shape[0]):
                if self.csv[j][i] == '?':
                    column_with_missing_info.append(i)
                    break

