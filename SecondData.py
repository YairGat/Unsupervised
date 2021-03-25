import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
from Data import Data


class SecondData(Data):
    classification = []

    def __init__(self):
        super().__init__('csv/diabetic_data.csv', ',')

    def _load_csv(self):
        number_of_rows = 101767
        random_list = np.random.choice(range(1, number_of_rows), size=15000, replace=False)
        random_list[0] = 0
        self.csv = self.csv[random_list]
        df = pd.DataFrame(data=self.csv[1:], columns=[title for title in self.csv[0]])
        classification_column = [2, 3]
        change_to_numeric = ['race',
                             'gender',
                             'age',
                             'weight',
                             'payer_code',
                             'medical_specialty',
                             'diag_1',
                             'diag_2',
                             'diag_3',
                             'max_glu_serum',
                             'A1Cresult',
                             'metformin',
                             'repaglinide',
                             'nateglinide',
                             'chlorpropamide',
                             'glimepiride',
                             'acetohexamide',
                             'glipizide',
                             'glyburide',
                             'tolbutamide',
                             'pioglitazone',
                             'rosiglitazone',
                             'acarbose',
                             'miglitol',
                             'troglitazone',
                             'tolazamide',
                             'examide',
                             'citoglipton',
                             'insulin',
                             'glyburide-metformin',
                             'glipizide-metformin',
                             'glimepiride-pioglitazone',
                             'metformin-rosiglitazone',
                             'metformin-pioglitazone',
                             'change',
                             'diabetesMed',
                             'readmitted']
        for i in change_to_numeric:
            temp_list = []
            for j in df[i]:
                if not temp_list.__contains__(j):
                    temp_list.append(j)
            temp_to_dict = self.convert_array_to_dict(temp_list)
            for j in range(0, len(df[i])):
                df[i][j] = temp_to_dict[df[i][j]]
        self.classification = []
        self.csv = pd.DataFrame(df).to_numpy()
        self.classification = self.csv[:, classification_column]
        self.csv = np.delete(self.csv, 2, 1)
        self.csv = np.delete(self.csv, 2, 1)
        self.classification = self.dimension_reduction_classification_to_1d()

    def _get_classification(self):
        return self.classification

    def convert_array_to_dict(self, arr):
        dict = {}
        for i in range(0, len(arr)):
            dict[arr[i]] = i
        return dict

    def dimension_reduction_classification_to_1d(self):
        pca = PCA(n_components=1)
        principal_components = pca.fit_transform(StandardScaler().fit_transform(self.classification))
        temp = np.zeros(len(principal_components))
        for i in range(0, len(principal_components)):
            temp[i] = principal_components[i][0]
        return temp

    def plot_optimal_clusters(self):
        self.k_means(24)
        self.gmm(24)
        self.fcm(4)
        self.k_means(4)
        self.k_means(4)

