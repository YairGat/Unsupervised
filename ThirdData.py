from sklearn.metrics import adjusted_mutual_info_score
from Data import Data
import numpy as np

# This class represents the third data set 'e-shop clothing 2008.csv'
class ThirdData(Data):
    classification = []

    def __init__(self):
        super().__init__('csv/e-shop clothing 2008.csv', ';')

    def help_list(self):
        list = self.csv[1:, 6]
        new_list = []
        for mem in list:
            if not new_list.__contains__(mem):
                new_list.append(mem)
        return new_list

    # Return the ascii value of element- helped function f
    @staticmethod
    def sort_by_first_char(element):
        return ord(element[0])

    @staticmethod
    def sort_by_numbers(element):
        number = int(element[1:])
        return number

    # Gets list and return dictionary with key that equals to indexs.
    @staticmethod
    def list_to_dictionary(list):
        index = 1
        d = dict()
        for mem in list:
            d[index] = mem
            index = index + 1
        return d

    def build_help_dictionary(self):
        l = self.help_list()
        l.sort(key=self.sort_by_numbers)
        l.sort(key=self.sort_by_first_char)
        return self.list_to_dictionary(l)

    def _get_classification(self):
        return self.classification[1:]

    # update the csv to numeric data.
    def _load_csv(self):
        self.classification = self.csv[:, 4]
        self.csv = np.delete(self.csv, 4, 1)
        d = self.build_help_dictionary()
        for i in range(1, len(self.csv[1:, 6]) + 1):
            update_value = self.get_key(self.csv[i][6], d)
            self.csv[i][6] = update_value

    @staticmethod
    def get_key(val, dict):
        for key, value in dict.items():
            if val == value:
                return key
        return "key doesn't exist"

    # If we would looking for the most reality solution, 47 was the optimal solution because we have 47
    # different country.
    def plot_optimal_clusters(self):
        self.k_means(8)
        self.gmm(4)
        self.fcm(4)
        self.k_means(4)
        self.k_means(4)


