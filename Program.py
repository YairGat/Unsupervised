import matplotlib.pyplot as plt

from FirstData import FirstData
from SecondData import SecondData
from ThirdData import ThirdData


def find_every_thing(data):
    # data.silhouette_all()
    optimal_fcm = data.get_optimal_number_of_clustering(data.fcm)
    optimal_kmeans = data.get_optimal_number_of_clustering(data.k_means)
    optimal_hierarchical = data.get_optimal_number_of_clustering(data.hierarchical)
    optimal_spectral = data.get_optimal_number_of_clustering(data.spectral)
    optimal_gmm = data.get_optimal_number_of_clustering(data.gmm)
    print("FCM- optimal number of clustering" + str(optimal_fcm))
    print("kmeans- optimal number of clustering" + str(optimal_kmeans))
    print("hierarchial- optimal number of clustering" + str(optimal_hierarchical))
    print("spectral- optimal number of clustering" + str(optimal_spectral))
    print("gmm- optimal number of clustering" + str(optimal_gmm))
    data.k_means(optimal_kmeans)
    data.gmm(optimal_gmm)
    data.fcm(optimal_fcm)
    data.hierarchical(optimal_hierarchical)
    data.spectral(optimal_spectral)
    print("AMI FUZZY: " + str(data.ami(data.fcm, optimal_fcm)))
    print("AMI HIERARCHIAL: " + str(data.ami(data.hierarchical, optimal_hierarchical)))
    print("AMI GMM: " + str(data.ami(data.gmm, optimal_gmm)))
    print("AMI KMEANS: " + str(data.ami(data.k_means, optimal_kmeans)))
    print("AMI SPECTRAL: " + str(data.ami(data.spectral, optimal_spectral)))


print('First Data------')
first_data = FirstData()
find_every_thing(first_data)
print('Second Data------')
second_data = SecondData()
find_every_thing(second_data)
print('Third Data-----')
third_data = ThirdData()
find_every_thing(third_data)
