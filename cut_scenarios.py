#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 11:50:04 2019

@author: dominik
"""
import os

import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

audio_durations = [154, 154, 119, 120, 167]

parser = lambda date: pd.datetime.strptime(date, '%d%m%Y-%H:%M:%S:%f')
target_dir = "./auswertung/results_cut"

def cut_tracking_into_scenarios(tracking_file, timestamp_file, test_id):
    timestamps = pd.read_csv(timestamp_file, header=None, names=["scenario", "timestamp"], parse_dates=["timestamp"], index_col="scenario")
    tracking_data = pd.read_csv(tracking_file, header=None, names=["timestamp", "x", "y", "z", "roll", "pitch", "yaw"], parse_dates=["timestamp"], date_parser=parser)
    
    scenario_dfs = []
#    
    for i in range(1, 6):
        start = timestamps.loc[i]
        start = start["timestamp"]
        df = tracking_data[(tracking_data["timestamp"] >= start) & (tracking_data["timestamp"] <= pd.to_datetime(pd.to_datetime(start)+pd.to_timedelta(audio_durations[i-1], unit="s")))]
        df["test_id"] = test_id
        df["scenario"] = i
        scenario_dfs.append(df)
#    
    return timestamps, tracking_data, scenario_dfs

def write_scenarios_to_single_files(scenario_dfs):
    for idx, scenario_df in enumerate(scenario_dfs):
        if scenario_df.empty:
            print("empty df!")
            continue
        test_id = scenario_df["test_id"].iloc[0]
        fn = "{}_{}.csv".format(test_id, idx+1)
        scenario_df.to_csv(os.path.join(target_dir, fn))
        
def get_dropout_information_from_scenario_df(scenario_df):
    dropout_info = {}
    dropouts = scenario_df["x"].isnull().sum()
    total_length = len(scenario_df)
    
    longest_dropout = 0
    current_longest_dropout = 0
    
    for x in scenario_df["x"]:
        if (pd.isna(x)):
            current_longest_dropout += 1
            if current_longest_dropout > longest_dropout:
                longest_dropout = current_longest_dropout
        else:
            current_longest_dropout = 0
                
    dropout_info["dropout_rate"] = dropouts/total_length
    dropout_info["longest_dropout"] = longest_dropout
    dropout_info["total_dropouts"] = dropouts
    
    return dropout_info    

def get_data_from_dir(directory):
    df = None
    timestamps = {}
    for root, dirs, files in os.walk(directory):
        timestamp_file = ""
        tracking_file = ""
        subject_id = None
        for fn in files:
            filepath = os.path.join(root, fn)
            subject_id = int(os.path.basename(root))
            print(root, subject_id)
            if "timestamps" in fn:
#                timestamp_df = pd.read_csv(filepath, header=None, names=["scenario", "timestamp"], parse_dates=["timestamp"], index_col="scenario")
#                timestamps[subject_id] = timestamp_df
                timestamp_file = filepath
            else: 
#                tracking_data = pd.read_csv(filepath, header=None, names=["timestamp", "x", "y", "z", "roll", "pitch", "yaw"], parse_dates=["timestamp"], date_parser=parser)
#                tracking_data["subject_id"] = int(subject_id)
#                if df is None:
#                    df = tracking_data
#                else: 
#                    df = pd.concat([df, tracking_data])
                tracking_file = filepath
        if subject_id is None:
            continue
        timestamps, tracking_data, scenario_dfs = cut_tracking_into_scenarios(tracking_file, timestamp_file, subject_id)
        write_scenarios_to_single_files(scenario_dfs)
        
                    
            
            
#    return df, timestamps

def calc_dropout_stats(df):
    total_frames = df.groupby("subject_id").count()["timestamp"]
    dropout_frames = total_frames - df.groupby("subject_id").count()["x"]
    dropout_rate = dropout_frames / total_frames
    
    dropout_frames = dropout_frames.reset_index()
    dropout_rate = dropout_rate.reset_index()
    
    return dropout_frames, dropout_rate

def linear_regression_dropoutrate_with_z(df):
    dropout_frames, dropout_rate = calc_dropout_stats(df)
    
    dropouts = np.array(dropout_rate[0]).reshape(-1, 1)
    zs = np.array(df.groupby("subject_id").mean()["z"]).reshape(-1, 1)
    
    regr = linear_model.LinearRegression()
    regr.fit(zs, dropouts)
    
    z_test = np.arange(1400, 2100, 1).reshape(-1,1)
    
    dropout_pred = regr.predict(z_test)
    
    plt.scatter(zs, dropouts)
    plt.plot(z_test, dropout_pred)
    
def set_scenario_ids(df, timestamps):
    for key, val in timestamps.items():
        for scenario in range(1,6):
            print(key, val, scenario)
            start = val.loc[[scenario],["timestamp"]]
            scenario_df = df[(df["subject_id"] == key) & (df["timestamp"] >= start) & (df["timestamp"] <= pd.to_datetime(pd.to_datetime(start)+pd.to_timedelta(audio_durations[scenario-1], unit="s")))]
            
            fn = f"{key}_{scenario}.csv"
            scenario_df.to_csv(os.path.join(target_dir, fn))
            
    return df
    
if __name__ == "__main__":
    result_dir = "./results"
    tracking_file = os.path.join(result_dir, "1.csv")
    timestamp_file = os.path.join(result_dir, "start_timestamps.csv")
    
    df, timestamps = get_data_from_dir(result_dir)
    
    df = set_scenario_ids(df, timestamps)
    
#    timestamp, tracking_data, scenario_dfs = cut_tracking_into_scenarios(tracking_file, timestamp_file, 1)
#    #write_scenarios_to_single_files(scenario_dfs)
#    for scenario_df in scenario_dfs:
#        dropout_info = get_dropout_information_from_scenario_df(scenario_df)
#        print(dropout_info)
