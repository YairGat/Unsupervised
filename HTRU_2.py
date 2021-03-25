from Data import Data


class AllUsersData(Data):
    classification = []

    def __init__(self):
        super().__init__('csv/HTRU_2.csv', ',')

    def _load_csv(self):
        self.classification = self.csv[:, 8]
        pass

all_user_data = AllUsersData()
all_user_data.dbscan()