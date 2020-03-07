import numpy as np
from pandas.api.types import is_numeric_dtype


class DataDimError(Exception):
    def __init__(self, expected, received):
        message = "".join(map(str, ["Invalid number of features, expected: ", expected, " received: ", received]))
        super(DataDimError, self).__init__(message)

class DataTargetMissmatch(Exception):
    def __init__(self, data, target):
        message = "".join(map(str, ["Number of data examples: ", data ," does not match target: ", target," examples"]))
        super(DataTargetMissmatch, self).__init__(message)

class DataTypeError(Exception):
    def __init__(self):
        message = "".join(map(str,["Invalid type of data, expected numerical."]))
        super(DataTypeError, self).__init__(message) 


class LinRegValidator:
    def __init__(self, n_features):
        self.n_features = n_features

    def validate_training(self, data, target):
        self.__validate_data(data)
        self.__validate_data_type(data)
        self.__validate_data_type(target)
        self.__check_if_data_and_target_match(data, target)

    def __validate_data(self, data):
        if data.shape[1] != self.n_features:
            raise DataDimError(data.shape[1], self.n_features)

    def __validate_data_type(self, data):
        if is_numeric_dtype(data) != True:
            raise DataTypeError()

    def __check_if_data_and_target_match(self, data, target):
        if data.shape[0] != target.shape[0]:
            raise DataTargetMissmatch(data.shape[0], target.shape[0])