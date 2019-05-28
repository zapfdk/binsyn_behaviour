#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 26 20:24:14 2019

@author: dominik
"""
import os
import datetime

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def prepare_angles_for_interp(df):
    yaw_complemented = np.rad2deg(np.unwrap(np.deg2rad(df["yaw"].dropna()), discont=np.pi/2))
    df["yaw"][~df["yaw"].isna()] = yaw_complemented.reshape(-1,1)
    return df

def get_tracking_files(result_dir):
    tracking_files = []
    for root, _, files in os.walk(result_dir):
        for fn in files:
            tracking_files.append(os.path.join(result_dir, fn))
            
    return tracking_files

def filter_list_by_scenario(files, scenario):
    return [x for x in files if (f"_{scenario}.csv" in x)]

def filter_list_by_id(files, id):
    return [x for x in files if (f"/{id}_" in x)]


def get_id_and_scenario_from_fn(fn):
    basename = os.path.splitext(os.path.basename(fn))[0]
    id = int(basename.split("_")[0])
    scenario = int(basename.split("_")[1])
    return (id, scenario)

def timeTicks(x, pos):
    seconds = x / 10**9 # convert nanoseconds to seconds
    # create datetime object because its string representation is alright
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    d = datetime.timedelta(seconds=seconds)
    return f"{int(minutes):02}:{int(seconds):02}"

def create_subplots_for_traces(scenario, nrows=5, ncols=5, polar=False, title=""): 
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True, subplot_kw=dict(polar=polar))
    fig.subplots_adjust(hspace=0, wspace=0)
    fig.suptitle(title)
    return fig, ax

def load_df_from_fn(fn):    
        df = pd.read_csv(fn, parse_dates=["timestamp"])
        df = df.sort_values(by="timestamp")
        df["timestamp"] = df["timestamp"] - df["timestamp"][0]
        
        # Little x offset due to difference between calibration and mid point
        x_offset = 0.8
        
        # Millimeter in Meter
        df["x"] = (df["x"]/1000)+x_offset
        df["y"] = df["y"]/1000
        
        return df
        
def calc_poly_area(x,y):
    """
    Calculate polygon area from given points, following this answer:
        https://stackoverflow.com/a/30408825/4906456
    Using the shoelace algorithm: 
        https://en.wikipedia.org/wiki/Shoelace_formula
    """
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))