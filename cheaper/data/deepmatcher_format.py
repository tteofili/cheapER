import os
import re

import pandas as pd

datadir = 'datasets' + os.sep + 'temporary' + os.sep


def tofiles(pt_train, pt_valid, name):
    pt_train = unflat(pt_train, 2)
    pt_valid = unflat(pt_valid, 2)
    names = []
    names.append('id')
    names.append('label')
    attr_per_tab = int((len(pt_valid[0]) - 2) / 2)
    for i in range(attr_per_tab):
        names.append('left_attr_' + str(i))
    for i in range(attr_per_tab):
        names.append('right_attr_' + str(i))

    df = pd.DataFrame(pt_train)
    if len(pt_train) > 0:
        cols = len(df.columns)
        for i in range(cols):
            if i > len(names):
                df.drop(df.columns[i], axis=1, inplace=True)
        df = df.dropna(axis=0,
                       how='any',
                       subset=None, )
        df.columns = names
    trainName = datadir + name + '_trainSim'+str(len(pt_train))+'.csv'
    os.makedirs(trainName[:trainName.rfind('/')], exist_ok=True)

    tr = df.astype(str).to_csv(trainName, index=False)

    df = pd.DataFrame(pt_valid)
    if len(pt_valid) > 0:
        cols = len(df.columns)
        for i in range(cols):
            if i > len(names):
                df.drop(df.columns[i], axis=1, inplace=True)
        df = df.dropna(axis=0,
                       how='any',
                       subset=None, )
        df.columns = names
    validName = datadir + name + '_validSim'+str(len(pt_train))+'.csv'
    os.makedirs(validName[:validName.rfind('/')], exist_ok=True)

    vd = df.astype(str).to_csv(validName, index=False)

    return trainName, validName


def unflat(data, target_idx, shrink=False):
    def cut_string(s):
        if len(s) >= 1000:
            return s[:1000]
        else:
            return s

    temp = []
    id = 0
    for r in data:
        t1 = r[0]
        t2 = r[1]
        lb = r[target_idx]
        if isinstance(lb, list):
            lb = lb[0]
        if (shrink):
            t1 = list(map(cut_string, t1))
            t2 = list(map(cut_string, t2))
        if len(t1) == len(t2):
            row = []
            row.append(id)
            row.append(lb)
            for a in t1:
                a = re.sub('[,\n\t\r;]', '', a)
                row.append(a)
            for a in t2:
                a = re.sub('[,\n\t\r;]', '', a)
                row.append(a)
            temp.append(row)
            id = id + 1

    return temp

