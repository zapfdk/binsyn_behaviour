#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 14:51:12 2019

@author: dominik
"""

import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib
from scipy import interpolate

from tqdm import tqdm
#...

from plot_stuff import get_tracking_files, result_dir, plot_dir, \
    filter_list_by_scenario, filter_list_by_id, get_id_and_scenario_from_fn
    
from visualization import handle_fig
    
x_offset = 0.8
metadaten_fn = "./probanden_metadaten.csv"

figure_dpi = 300
axis_label_font_size = 18
subplot_title_font_size = 16
tick_label_size = 15


matplotlib.rc('xtick', labelsize=10) 
matplotlib.rc('ytick', labelsize=10) 


# Little x offset due to difference between calibration and mid point
x_offset = 0.8

metadata = pd.read_csv(metadaten_fn)    

def load_all_files_into_df(files):
    df = None
    for fn in files:
        if df is None:
            df = pd.read_csv(fn, parse_dates=["timestamp"])
        df = df.append(pd.read_csv(fn, parse_dates=["timestamp"]))
    
    df = df.sort_values(by="timestamp")
    
    return df

def get_longest_dropout_streak(df):
    longest_streak = 0
    current_streak = 0
    
    last_row_with_data = None
    currently_nan = False
    
    data_differences = []
    
    for _, row in df.iterrows():
        if pd.isna(row["x"]):
            current_streak += 1
            currently_nan = True
        else: 
            current_streak = 0
            
            if currently_nan and last_row_with_data is not None:
                diff = row-last_row_with_data
                diff["distance"] = np.sqrt(diff["x"]**2 + diff["y"]**2)
                diff["yaw"] = get_abs_angle_diff(row["yaw"], last_row_with_data["yaw"])
                data_differences.append(diff)
                
            
            last_row_with_data = row
            currently_nan = False
            
        if current_streak > longest_streak:
            longest_streak = current_streak
    
    return longest_streak, data_differences 

def get_data_diff_stats(data_differences):
    diff_df = pd.DataFrame(data_differences)

    stats = {"maxs": diff_df.abs().max(axis=0),
     "mins": diff_df.abs().min(axis=0), 
     "means": diff_df.abs().mean(axis=0),
     "medians": diff_df.abs().median(axis=0)
     }
    
    stats["means"]["timestamp"] = pd.to_timedelta(diff_df["timestamp"].values.astype(np.int64).mean())
    #stats["medians"]["timestamp"] = pd.to_timedelta(diff_df["timestamp"].values.astype(np.int64).median())
    
    
    return stats

def plot_dropout_rates(dropout_stats):
    rates = []
    for scenario in range(1,6):
        rates.append(dropout_stats["scenario"][scenario]["dropout_rate"])
    
    
    fig = plt.figure()
    ax = fig.gca()
    #...
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlim([0.5,5.5])
    ax.set_ylim([0,1])
    plt.scatter(range(1,6), rates)
    
    ax.set_xlabel("Szenario", fontsize=axis_label_font_size)
    ax.set_ylabel("Aussetzrate", fontsize=axis_label_font_size)
    
    handle_fig(fig, f"aussetzrate_scenario.png")
    
def print_relevant_dropout_stats(dropout_stats):
    for scenario in range(1,6):
        print(f"\nData for scenario {scenario}")
        scenario_data = dropout_stats["scenario"][scenario]["diff_stats"]
        maxs = scenario_data["maxs"]
        means = scenario_data["means"]
        
        print("L_max", maxs["timestamp"])
        print("L_mean", pd.to_timedelta(means["timestamp"]))
        print("D_pos,max", maxs["distance"])
        print("D_pos,mean", means["distance"])
        print("D_gier,max", maxs["yaw"])
        print("D_gier,mean", means["yaw"])
        
        
def get_abs_angle_diff(a1, a2):
    angle_diff = 180 - abs(abs(a1-a2) - 180)
    return angle_diff

def test_angle_interp():
    """
    https://stackoverflow.com/a/27303263/4906456
    """
    df = pd.DataFrame({"yaw": [0,5,np.NaN, np.NaN, 120, 190, np.NaN, 220, 355, np.NaN, np.NaN, 5, np.NaN, 359]})
    yaw_comp = np.rad2deg(np.unwrap(np.deg2rad(df["yaw"].dropna())))
    df["yaw"][~df["yaw"].isna()] = yaw_comp.reshape(-1,1)
    df["yaw_interpolated"] = df["yaw"].interpolate() % 360
    plt.plot(df["yaw"])
    plt.figure()
    plt.plot(df["yaw_interpolated"])
    
def prepare_angles_for_interp(df):
    yaw_complemented = np.rad2deg(np.unwrap(np.deg2rad(df["yaw"].dropna())))
    df["yaw"][~df["yaw"].isna()] = yaw_complemented.reshape(-1,1)
    return df
    
#print_relevant_dropout_stats(dropout_stats)
    
def dropout_heatmap(df, df_interpolated, scenario):
#    df = prepare_angles_for_interp(df)
#    
#    df_interpolated = df.interpolate(limit_direction="both")
        
    dropout_locations = df_interpolated[df["x"].isna()]
    
    H_total, xedges, yedges = np.histogram2d(df_interpolated["x"], df_interpolated["y"], bins=50, range=[[-2,2], [-2,2]])
    H_dropouts, xedges, yedges = np.histogram2d(dropout_locations["x"], dropout_locations["y"], bins=50, range=[[-2,2], [-2,2]])
    
    H_normalized = H_dropouts/H_total   
    
    np.nan_to_num(H_normalized, copy=False)
    
    fig = plt.figure()
    #plt.title(f"Heatmap der Aussetzer für Szenario {scenario}")
    plt.imshow(H_normalized, interpolation='bilinear', origin='upper', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    #plt.hist2d(dropout_locations["x"], dropout_locations["y"], range=[[-2000,2000],[-2000,2000]], bins=20)
    
    ax = fig.gca()
    ax.set_xlabel("X-Position", fontsize=axis_label_font_size)
    ax.set_ylabel("Y-Position", fontsize=axis_label_font_size)
    
    handle_fig(fig, f"aussetzer_heatmap.png")
    
def calculate_stats_for_dropout():
    dropout_stats = {"scenario":{},
                     "id": {}}
    
    for scenario in range(1, 6):
        print(f"Calculating dropout for scenario {scenario}")
        dropout_stats["scenario"][scenario] = {}
        scenario_files = filter_list_by_scenario(tracking_files, scenario)
        
        total_dropouts = 0
        total_length = 0
        data_differences = []  
        longest_streaks = []
        
        for fn in tqdm(scenario_files):
            df = pd.read_csv(fn, parse_dates=["timestamp"])
            df = df.sort_values(by="timestamp")
            
            df["yaw"][df["yaw"] <0] += 360
            
            dropouts = df.isnull().sum(axis=0)["x"]
            length = len(df)
            
            total_dropouts += dropouts
            total_length += length
            
            longest_streak, data_differences = get_longest_dropout_streak(df)
            longest_streaks.append(longest_streak)
            data_differences.extend(data_differences)           
                    
        dropout_stats["scenario"][scenario]["dropout_rate"] = total_dropouts/total_length
        
        longest_streak, data_differences = get_longest_dropout_streak(df)
        dropout_stats["scenario"][scenario]["longest_dropout_nb"] = max(longest_streaks)
        dropout_stats["scenario"][scenario]["data_differences"] = data_differences
        
        diff_stats = get_data_diff_stats(data_differences)
        dropout_stats["scenario"][scenario]["diff_stats"] = diff_stats
       
    return dropout_stats
    #print_relevant_dropout_stats(dropout_stats)
    
def plot_dropout_heatmaps():
    for scenario in [5]:
        print(f"Drawing heatmap for scenario {scenario}")
        #dropout_stats["scenario"][scenario] = {}
        scenario_files = filter_list_by_scenario(tracking_files, scenario)
        
        dfs = []
        dfs_interpolated = []
        
        for scenario_file in tqdm(scenario_files):            
            scenario_df = pd.read_csv(scenario_file, parse_dates=["timestamp"])
            scenario_df["x"] = (scenario_df["x"]/1000)+x_offset
            scenario_df["y"] = scenario_df["y"]/1000
            dfs.append(scenario_df)
            scenario_df_interpolated = scenario_df.interpolate(limit_direction="both")
            dfs_interpolated.append(scenario_df_interpolated)
        df = pd.concat(dfs)
        df_interpolated = pd.concat(dfs_interpolated)
        
        dropout_heatmap(df, df_interpolated, scenario)  
        
def plot_dropout_per_height():
    avg_zs = []
    droprates = []
    
    fig = plt.figure()
    
    for id in range(1,24):
        id_files = filter_list_by_id(tracking_files, id)
        
        id_df = pd.DataFrame()
        for id_file in id_files:
            df = pd.read_csv(id_file, parse_dates=["timestamp"])
            df = df.sort_values(by="timestamp")
            
            id_df = id_df.append(df)        
        
        dropouts = id_df.isnull().sum(axis=0)["z"]
        avg_zs.append(float(id_df["z"].mean()))
        droprates.append(dropouts/len(id_df))
    
    plt.scatter(np.array(avg_zs)/1000, droprates)
        
    ax=plt.gca()
    ax.set_ylabel("Durchschnittliche Ausfallrate der Probanden", fontsize=axis_label_font_size)
    ax.set_xlabel("Durchschnittliche z-Position der Probanden in m", fontsize=axis_label_font_size)
    
    handle_fig(fig, f"aussetzer_z_position.png")
    
    return avg_zs, droprates

if __name__ == "__main__":
    tracking_files = get_tracking_files()
    plot_dropout_per_height()
    plot_dropout_heatmaps()
    plot_dropout_rates(calculate_stats_for_dropout())
        
