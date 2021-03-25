import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.cluster import adjusted_mutual_info_score

from Data import Data


class FirstData(Data):
    classification = []

    def __init__(self):
        super().__init__('csv/online_shoppers_intention.csv', ',')

    def _load_csv(self):
        month = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'June': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10,
                 'Nov': 11, 'Dec': 12}
        month_column = 10
        VisitorType_dict = {'New_Visitor': 0, 'Returning_Visitor': 1, 'Other': 2}
        VisitorType_column = 15
        boolean_dict = {'FALSE': 0, 'TRUE': 1}
        weekend_column = 16
        revenue_column = 17
        for i in range(1, len(self.csv[1:]) + 1):
            self.csv[i][month_column] = month[self.csv[i][month_column]]
            self.csv[i][VisitorType_column] = VisitorType_dict[self.csv[i][VisitorType_column]]
            self.csv[i][weekend_column] = boolean_dict[self.csv[i][weekend_column]]
            self.csv[i][revenue_column] = boolean_dict[self.csv[i][revenue_column]]

        self.classification = self.csv[:, [VisitorType_column, weekend_column, revenue_column]]
        self.classification_label_1 =  self.classification[:,0]

        self.csv = np.delete(self.csv, 15, 1)
        self.csv = np.delete(self.csv, 15, 1)
        self.csv = np.delete(self.csv, 15, 1)
        self.classification = self.dimension_reduction_classification_to_1d()


    def dimension_reduction_classification_to_1d(self):
        pca = PCA(n_components=1)
        principal_components = pca.fit_transform(StandardScaler().fit_transform(self.classification[1:]))
        temp = np.zeros(len(principal_components))
        for i in range(0, len(principal_components)):
            temp[i] = principal_components[i][0]
        return temp

    def ami_label_one(self):
        return adjusted_mutual_info_score(self.gmm(3, plot=False),
                                          self.classification_label_1)
    def ami_label_two(self):
        return adjusted_mutual_info_score(self.gmm(3, plot=False),
                                          self.classification_label_2)
    def ami_label_three(self):
        return adjusted_mutual_info_score(self.gmm(3, plot=False),
                                          self.classification_label_3)

fir = FirstData()
fir.get_clustered_data_without_anomaly_points_by_silhouette(fir.k_means(5, plot=False))