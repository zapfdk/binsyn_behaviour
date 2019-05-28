#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 12:14:26 2019

@author: dominik
"""

import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

datapath = "./probanden_metadaten.csv"

def gaussian(x,x0,sigma):
  return np.exp(-np.power((x - x0)/sigma, 2.)/2.)

if __name__ == "__main__":
    data = pd.read_csv(datapath)
    data = data[~data.isna().any(axis=1)]
    
    nb_samples = len(data)
    
    data_stats = {"Alter":{},
                  "Geschlecht": {},
                  "EingeschHoer": {},
                  "SchonHoertest": {},
                  "ErfBinSyn": {},
                  "KenntZielperson": {},
                  }
    
    data_stats["Alter"]["mean"] = data["Alter"].mean()
    data_stats["Alter"]["std"] = data["Alter"].std()
    data_stats["Alter"]["median"] = data["Alter"].median()
    data_stats["Alter"]["min"] = data["Alter"].min()
    data_stats["Alter"]["max"] = data["Alter"].max()
    
    data_stats["Geschlecht"]["m_total"] = data["Geschlecht"].value_counts()["m"]
    data_stats["Geschlecht"]["f_total"] = data["Geschlecht"].value_counts()["f"]
    data_stats["Geschlecht"]["m_percentage"] = data["Geschlecht"].value_counts()["m"]/nb_samples
    data_stats["Geschlecht"]["f_percentage"] = data["Geschlecht"].value_counts()["f"]/nb_samples
    
    for key in ["EingeschHoer", "SchonHoertest", "ErfBinSyn", "KenntZielperson"]:
        data_stats[key]["total"] = data[key].sum()
        data_stats[key]["percentage"] = data[key].sum()/nb_samples
        
    x = np.arange(min(data["Alter"]), max(data["Alter"]), 0.01)
    y = gaussian(x, data_stats["Alter"]["mean"], data_stats["Alter"]["std"])
        
    plt.hist(data["Alter"], bins=range(int(min(data["Alter"])), int(max(data["Alter"]))+1, 1))
    plt.plot(x, y*5)
        
    
    
    
    
    