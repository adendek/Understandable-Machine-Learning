import numpy as np
from pandas.api.types import is_numeric_dtype


class TargetValueError(Exception):
    def __init__(self, received, expected):
        message = "".join(map(str,["Invalid number of target classes, received: ", received ," expected: ", expected]))
        super(TargetValueError, self).__init__(message)


class DataDimError(Exception):
    def __init__(self, expected, received):
        message = "".join(map(str, ["Invalid number of features, expected: ", expected, " received: ", received]))
        super(DataDimError, self).__init__(message)

class DataTargetMissmatch(Exception):
    def __init__(self, data, target):
        message = "".join(map(str, ["Number of data examples: ", data ," does not match target: ", target," examples"]))
        super(DataTargetMissmatch, self).__init__(message)
        
class TargetDataCatError(Exception):
    def __init__(self, target):
        message = "".join(map(str,["Invalid type of target classes, received: ", target ," expected cat {0, 1} "]))
        super(TargetDataCatError, self).__init__(message)
        
class DataTypeError(Exception):
    def __init__(self):
        message = "".join(map(str,["Invalid type of data, expected numerical."]))
        super(DataTypeError, self).__init__(message) 


class LogRegValidator:
    def __init__(self, n_features, n_classes):
        self.n_classes = n_classes
        self.n_features = n_features

    def validate_training(self, data, target):
        self.__validate_target(target)
        self.__validate_target_type(target)
        self.__validate_data(data)
        self.__validate_data_type(data)
        self.__check_if_data_and_target_match(data, target)

    def __validate_target(self, target):
        if np.unique(target).shape[0] > self.n_classes:
            raise TargetValueError(np.unique(target).shape[0], self.n_classes)

    def __validate_target_type(self, target):
        if set(target) != {0, 1}:
            raise TargetDataCatError(set(target))
            
    def __validate_data(self, data):
        if data.shape[1] != self.n_features:
            raise DataDimError(data.shape[1], self.n_features)
            
    def __validate_data_type(self, data):
        if set([is_numeric_dtype(data[e]) for e in data.columns]) != {True}:
            raise DataTypeError()

    def __check_if_data_and_target_match(self, data, target):
        if data.shape[0] != target.shape[0]:
            raise DataTargetMissmatch(data.shape[0], target.shape[0])
