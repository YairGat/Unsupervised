import HTRU_2
import AllUsersData


def do_all(data_name, file_name, anomaly=False):
    f = open(file_name, 'a')
    f.write('Anomaly points removed: ' + str(anomaly) + '\n')
    kmeans = data_name.get_optimal_number_of_clusters(data_name.k_means, anomaly=anomaly)
    f.write('K-Means: the optimal number of clusters: ' + str(kmeans) + '\n')
    gmm = data_name.get_optimal_number_of_clusters(data_name.gmm, anomaly=anomaly)
    f.write('GMM: the optimal number of clusters: ' + str(gmm) + '\n')
    fcm = data_name.get_optimal_number_of_clusters(data_name.fcm, anomaly=anomaly)
    f.write('FCM: the optimal number of clusters: ' + str(fcm) + '\n')
    spectral = data_name.get_optimal_number_of_clusters(data_name.spectral, anomaly=anomaly)
    f.write('Spectral: the optimal number of clusters: ' + str(spectral) + '\n')
    hierarchical = data_name.get_optimal_number_of_clusters(data_name.hierarchical, anomaly=anomaly)
    f.write('Hierarchical: the optimal number of clusters: ' + str(hierarchical) + '\n')
    ami = data_name.ami(data_name.k_means, optimal_number_of_clustering=kmeans)
    f.write('The ami of K-Means: ' + str(ami) + '\n')
    ami = data_name.ami(data_name.k_means, optimal_number_of_clustering=kmeans, anomaly=anomaly)
    f.write('The ami of K-Means: ' + str(ami) + '\n')
    f.write('The ami of GNN: ' + str(
        data_name.ami(data_name.gmm, optimal_number_of_clustering=gmm, anomaly=anomaly)) + '\n')
    f.write('The ami of FCM: ' + str(
        data_name.ami(data_name.fcm, optimal_number_of_clustering=fcm, anomaly=anomaly)) + '\n')
    f.write('The ami of Spectral: ' + str(
        data_name.ami(data_name.spectral, optimal_number_of_clustering=spectral, anomaly=anomaly)) + '\n')
    f.write('The ami of Hierarchical: ' + str(data_name.ami(data_name.hierarchical,
                                                            optimal_number_of_clustering=hierarchical,
                                                            anomaly=anomaly)) + '\n')


print('################HTRU2################# - with anomaly')
do_all(HTRU_2.HTRU_2(), file_name='Results/HTRU_2', anomaly=False)
print('################HTRU2################# - without anomaly')
do_all(HTRU_2.HTRU_2(), file_name='Results/HTRU_2', anomaly=True)
print('################All User Data################# - with anomaly')
do_all(AllUsersData.AllUsersData(), file_name='Results/allUsers', anomaly=False)
print('################All User Data################# - without anomaly')
do_all(AllUsersData.AllUsersData(), file_name='Results/allUsers', anomaly=True)
