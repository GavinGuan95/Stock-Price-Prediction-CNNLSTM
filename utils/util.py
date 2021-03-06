import json
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
import numpy as np


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.conf_mtx = np.array([[0,0],[0,0]])
        self.comp_mtx = None
        self.sign_mtx = None
        self.reset()
        
    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

        self.conf_mtx = np.array([[0, 0], [0, 0]])
        self.comp_mtx = None
        self.sign_mtx = None


    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)

        if key =='confusion_matrix':
            self.conf_mtx+=value
        elif key == 'MAPE':
            if self.comp_mtx is None:
                self.comp_mtx = value
            else:
                self.comp_mtx = np.concatenate((self.comp_mtx, value),axis=0)
        elif key == 'roc_auc':
            if self.sign_mtx is None:
                self.sign_mtx = value
            else:
                self.sign_mtx = np.concatenate((self.sign_mtx, value), axis=0)
        else:
            self._data.total[key] += value * n
            self._data.counts[key] += n
            self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def get_conf_mtx(self):
        return self.conf_mtx

    def get_comp_mtx(self):
        return self.comp_mtx

    def get_sign_mtx(self):
        return self.sign_mtx

    def result(self):
        return dict(self._data.average)

