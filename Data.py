import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib as plt
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_samples
import skfuzzy as fuzz
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
from scipy import stats
# outlier/anomaly detection
from sklearn.neighbors import LocalOutlierFactor


# Data class represents a base class for each data.
class Data:
    NUMBER_OF_SAMPLES = 8000

    # _init_ Data with .
    def __init__(self, csv_path, delimiter):
        # csv name.
        self.csv_path = csv_path
        # Sign that separator between column.
        self.delimiter = delimiter
        # hold the csv content.
        self.csv = np.genfromtxt(csv_path, delimiter=delimiter,
                                 encoding='utf8', dtype=np.str)
        # change no numbers places in the csv to be numbers.
        self._load_csv()
        # hold the data after dimension reduction.
        self.principalDfOriginal = self.dimension_reduction()
        self.principalDf = self.get_sample_from_data(0)
        # hold the classification column with same index of the sample
        self.classification = self._get_classification()[self.principalDf.index]
        csv_name = csv_path.split('/')[1].split('.')[0]
        self.file_path = open('Results\\' + csv_name, 'a')
        self.file_path.write(csv_name + '\n')

    def _get_classification(self):
        return self.classification

    def dimension_reduction(self):
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(StandardScaler().fit_transform(self.get_content()))
        principal_df = pd.DataFrame(data=principal_components,
                                    columns=['principal component 1', 'principal component 2'])
        return principal_df

    def get_principal_df(self):
        return self.principalDf

    # Every data update the csv differently.
    def _load_csv(self):
        pass

    def get_titles(self):
        return self.csv[0]

    def get_content(self):
        return self.csv[1:]

    def get_sample_from_data(self, random_state):
        return self.principalDfOriginal.sample(n=self.NUMBER_OF_SAMPLES, random_state=random_state)

    def silhouette(self, clustering_algorithm, title='', plot=True, i=0, anomaly=False):
        if anomaly:
            data = self.get_data_without_anomaly_points()
        elif i == 0:
            data = self.principalDf
        else:
            data = self.get_sample_from_data(i)
        # This list will hold the silhouette scores.
        silhouette_list = []
        # Range of silhouette.
        r = range(2, 8)
        for k in r:
            labels = self.cluster_labels(clustering_algorithm, k, i, anomaly=anomaly)
            silhouette_list.append(silhouette_score(data, labels))
        if not plot:
            return silhouette_list
        else:
            plt.plot(r, silhouette_list)
            plt.xlabel('Number of clusters')
            plt.ylabel('Silhouette score')
            plt.title('Silhouette Method For Optimal number of clusters-' + title)
            plt.show()

    ###### anomaly detection #######
    def LOF(self):
        df = self.principalDf
        # model specification
        model1 = LocalOutlierFactor(n_neighbors=2, metric="manhattan", contamination=0.02)
        # model fitting
        y_pred = model1.fit_predict(df)
        # filter outlier index
        outlier_index = np.where(y_pred == -1)  # negative values are outliers and positives inliers
        # filter outlier values
        outlier_values = df.iloc[outlier_index]
        return [df.iloc, y_pred]

    def get_index_without_anomaly_points(self):
        lof = self.LOF()[1]
        return np.where(lof == 1)

    def get_data_without_anomaly_points(self):
        core_points = self.LOF()[0][self.get_index_without_anomaly_points()]
        return core_points

    def get_classification_without_anomaly_points(self):
        return self._get_classification()[self.get_index_without_anomaly_points()]

    ###### CLUSTERING ALGHORITHM ###############
    def dbscan(self, plot=True, i=0):
        if i == 0:
            data_to_cluster = self.principalDf
        else:
            data_to_cluster = self.get_sample_from_data(i)
        model = DBSCAN(eps=0.3, min_samples=40).fit(data_to_cluster)

        if plot:
            self.cluster_plot(data_to_cluster, 'DBSCAN', model.labels_)
        else:
            return model.labels_

    def k_means(self, k, plot=True, i=0, anomaly=False):
        if anomaly:
            data = self.get_data_without_anomaly_points()
        elif i == 0:
            data = self.principalDf
        else:
            data = self.get_sample_from_data(i)
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data)
        labels = kmeans.predict(data)
        if plot:
            self.cluster_plot(data, 'K-MEAN', labels)
        return labels

    def gmm(self, k, plot=True, i=0, anomaly=False):
        if anomaly:
            data = self.get_data_without_anomaly_points()
        elif i == 0:
            data = self.principalDf
        else:
            data = self.get_sample_from_data(i)
        gmm = GaussianMixture(n_components=k).fit(data)
        labels = gmm.predict(data)
        if plot:
            self.cluster_plot(data, 'GMM', labels)
        return labels

    def hierarchical(self, k, plot=True, i=0, anomaly=False):
        if anomaly:
            data = self.get_data_without_anomaly_points()
        elif i == 0:
            data = self.principalDf
        else:
            data = self.get_sample_from_data(i)
        hierarchical = AgglomerativeClustering(n_clusters=k)
        hierarchical.fit(data)
        labels = hierarchical.labels_
        if plot:
            self.cluster_plot(data, 'Hierarchical', labels)
        return labels

    def spectral(self, k, plot=True, i=0, anomaly=False):
        if anomaly:
            data = self.get_data_without_anomaly_points()
        elif i == 0:
            data = self.principalDf
        else:
            data = self.get_sample_from_data(i)
        clustering = SpectralClustering(n_clusters=k, assign_labels="discretize", random_state=0)
        labels = clustering.fit_predict(data)
        if plot:
            self.cluster_plot(data, 'Spectral', labels)
        return labels

    def fcm(self, k, plot=True, i=0, anomaly=False):
        if anomaly:
            data = self.get_data_without_anomaly_points()
        elif i == 0:
            data = self.principalDf
        else:
            data = self.get_sample_from_data(i)
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(data.T, k, 2, error=0.005, maxiter=1000)
        labels = np.argmax(u, axis=0)
        if plot:
            self.cluster_plot(data, 'FCM', labels)
        return labels

    ###### END OF CLUSTERING ###########

    # Get get data and labels after clustering by 'title' method and plot.
    def cluster_plot(self, data, title, labels):
        plt.scatter(data['principal component 1'], data['principal component 2'], c=labels,
                    s=40)
        plt.title(title)
        plt.xlabel('principal component 1')
        plt.ylabel('principal component 2')
        # plt.legend()
        plt.show()

    # Get clustering method and return clustered data.
    def cluster_labels(self, cluster_type, k, i=0, anomaly=False):
        return cluster_type(k, plot = False, i=i, anomaly=anomaly)

    def get_silhouette_list(self, clustering_method, anomaly=False):
        silhouette = []
        r = np.random.choice(range(1, 100), 5, replace=False)
        for i in range(0, 4):
            silhouette.append(self.silhouette(clustering_method, plot=False, i=r[i], anomaly=anomaly))
        return np.transpose(silhouette)

    def t_test_(self, silhouette_list):
        best_cluster = silhouette_list[0]
        # In the second data the first sillhoute value is 9 while in the other is 2.
        start = 2
        best_number_of_clusters = 2
        sig = 0.05
        t = 2
        for i in range(len(silhouette_list)):
            self.file_path.write("The average silhouette score of " + str(t) + " clusters is: " + str(
                np.mean(silhouette_list[i])) + '\n')
            t = t + 1
        for i in range(0, len(silhouette_list)):
            t, p = stats.ttest_ind(best_cluster, silhouette_list[i], equal_var=False)
            if np.mean(best_cluster) > np.mean(silhouette_list[i]):
                p = p / 2
            else:
                p = 1 - p / 2
            clusters = i + start
            self.file_path.write(
                'H0: ' + str(best_number_of_clusters) + ' is more accurate then ' + str(clusters) + 'is: ' + str(
                    p) + '.\n')
            if float(p) > sig:
                best_cluster = silhouette_list[i]
                best_number_of_clusters = i + start
        return best_number_of_clusters

    def get_optimal_number_of_clusters(self, method, anomaly=False):
        if method == self.k_means:
            self.file_path.write('K-Means:\n')
        elif method == self.gmm:
            self.file_path.write('GMM:\n')
        elif method == self.fcm:
            self.file_path.write('FCM:\n')
        elif method == self.hierarchical:
            self.file_path.write('Hierarchical:\n')
        elif method == self.spectral:
            self.file_path.write('Spectral:\n')
        else:
            self.file_path.write('Other\n')
        return self.t_test_(self.get_silhouette_list(method, anomaly=anomaly))

    ######## Mutual Information ###########
    def ami(self, method, optimal_number_of_clustering=0, anomaly=False):
        if anomaly:
            classification = self.get_classification_without_anomaly_points()
        else:
            classification = self._get_classification()
        if optimal_number_of_clustering == 0:
            optimal_number_of_clustering = self.get_optimal_number_of_clusters(method)
        else:
            optimal_number_of_clustering = optimal_number_of_clustering
        return adjusted_mutual_info_score(method(optimal_number_of_clustering, plot=False, anomaly=anomaly),
                                          classification)

    def plot_optimal_clusters(self):
        pass

    def plot_all_silhouette(self):
        silhouette_kmeans = self.silhouette(self.k_means, "bla", plot=False)
        silhouette_gmm = self.silhouette(self.gmm, "bla", plot=False)
        silhouette_fcm = self.silhouette(self.fcm, "fcm", plot=False)
        silhouette_spectral = self.silhouette(self.spectral, "bla", plot=False)
        silhouette_hierarcial = self.silhouette(self.hierarchical, "bla", plot=False)
        if self.csv_path == 'diabetic_data.csv':
            r = range(5, 25)
        else:
            r = range(2, 15)
        kmeans, = plt.plot(r, silhouette_kmeans, label='Kmeans')
        gmm, = plt.plot(r, silhouette_gmm, label='GMM')
        fcm, = plt.plot(r, silhouette_fcm, label='FCM')
        spectral, = plt.plot(r, silhouette_spectral, label='Spectral')
        hierarcial, = plt.plot(r, silhouette_hierarcial, label='Hierarcial')
        plt.legend(handles=[kmeans, gmm, fcm, spectral, hierarcial])
        plt.show()
