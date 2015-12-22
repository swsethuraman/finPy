'''
Created on Jul 10, 2014

@author: SSethuraman
'''
import useful

class DataSet(useful.Observable):
    def __init__(self, name):
        super(DataSet, self).__init__()
        self.name       = name
        self.data       = {}
        self.version    = 1
        pass
    def add(self, keys, dicts):
        i = 0
        for k in keys:
            self.data.update({k   :   dicts[i]})
            i = i + 1
        self.version = self.version + 1
        self.notify()
        pass
    