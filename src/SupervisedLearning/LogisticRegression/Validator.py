import numpy as np


class TargetValueError(Exception):
    def __init__(self, seen, expected):
        message = "".join(map(str,["Invalid nb of target classes, seen: ",seen ," expected ", expected]))
        super(TargetValueError, self).__init__(message)


class DataDimError(Exception):
    def __init__(self, seen, expected):
        message = "".join(map(str, ["Invalid number of features, seen: ", seen," expected ", expected]))
        super(DataDimError, self).__init__(message)

class DataTargetMissmatch(Exception):
    def __init__(self, data, target):
        message = "".join(map(str, ["Number of data examples ", data ," does not match target ", target," examples"]))
        super(DataTargetMissmatch, self).__init__(message)


class LogRegValidator:
    def __init__(self, n_features, n_classes):
        self.n_classes = n_classes
        self.n_features = n_features

    def validate_training(self, data, target):
        self.__validate_target(target)
        self.__validate_data(data)
        self.__check_if_data_and_target_match(data, target)

    def __validate_target(self, target):
        if np.unique(target).shape[0] > self.n_classes:
            raise TargetValueError(np.unique(target).shape[0], self.n_classes)

    def __validate_data(self, data):
        if data.shape[0] != self.n_features:
            raise DataDimError(data.shape[0], self.n_features)

    def __check_if_data_and_target_match(self, data, target):
        if data.shape[1] != target.shape[0]:
            raise DataTargetMissmatch(data.shape[1], target.shape[0])
