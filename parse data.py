# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 09:11:29 2017

@author: jtay
"""

import pandas as pd




# Madelon
madX1 = pd.read_csv('./madelon_train.data',header=None,sep=' ')
madX2 = pd.read_csv('./madelon_valid.data',header=None,sep=' ')
madX = pd.concat([madX1,madX2],0).astype(float)
madY1 = pd.read_csv('./madelon_train.labels',header=None,sep=' ')
madY2 = pd.read_csv('./madelon_valid.labels',header=None,sep=' ')
madY = pd.concat([madY1,madY2],0)
madY.columns = ['Class']
mad = pd.concat([madX,madY],1)
mad = mad.dropna(axis=1,how='all')
mad.to_hdf('datasets.hdf','madelon',complib='blosc',complevel=9)