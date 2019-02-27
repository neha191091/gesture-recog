'''
Configuration File
'''

from enum import Enum

DataDir = 'data_set'
Url = 'http://zhen-wang.appspot.com/rice/files/uwave/uWaveGestureLibrary.zip'
DataZip = DataDir + '/uWaveGestureLibrary.zip'

DirPrefixTemplate = DataDir+'/'+'U{:d} ({:d})'+'/'
FileTemplate = DataDir+'/'+'U{:d} ({:d})'+'/'+'{:s}_Template_Acceleration{:d}-{:d}.txt'


class Num(Enum):
    users = 8
    days = 7
    classes = 8
    repeat = 10


class SplitType(Enum):
    train = 'train'
    val = 'val'
    test = 'test'


class ModelType(Enum):
    LogisticRegressionNumpy = 'LogisticRegressionNumpy'
    LogisticRegressionSklearn = 'LogisticRegressionSklearn'
    BidirectionalRNNTf = 'BidirectionalRNNTf'
    BidirectionalRNNKeras = 'BidirectionalRNNKeras'

