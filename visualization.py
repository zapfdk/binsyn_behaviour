#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 26 20:24:40 2019

@author: dominik
"""
import os
import itertools
import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mt
    
from scipy.spatial import ConvexHull
from tqdm import tqdm
from detect_peaks import detect_peaks
from statsmodels.tsa import stattools
import seaborn as sns

from utils import get_tracking_files, prepare_angles_for_interp, filter_list_by_scenario,\
    filter_list_by_id, get_id_and_scenario_from_fn, timeTicks, create_subplots_for_traces,\
    load_df_from_fn, calc_poly_area


result_dir = "./auswertung/results_cut"
plot_dir = "./auswertung/plots"
metadaten_fn = "./probanden_metadaten.csv"

save_figs = True 
close_figs = True 
figure_dpi = 300
axis_label_font_size = 18
subplot_title_font_size = 16
tick_label_size = 15
speed_subplot_title = 12


matplotlib.rc('xtick', labelsize=10) 
matplotlib.rc('ytick', labelsize=10) 


tracking_files = get_tracking_files(result_dir)
metadata = pd.read_csv(metadaten_fn)    

#
time_map = {1: [], 2: [], 3:[], 4:[], 5:[]}
segment_map = {
        1: [1,3,4,6,2,5,4,2,3,6,5,1],
        2: [2,4,6,5,1,3,1,3,2,6,4,5],
        3: [(0,5), (5,10),(10, 16), (16,22),(22, 27), (27,33),(33, 38), (38,44),(44, 49), (49,55), (55, 59), (59,66),(66, 71), (71,77),(77, 81), (81,88),(88, 93), (93,99),(99, 103), (103,106), (106, 115)],
        4: [(2,(0,25)), (5,(27,35)), (6,(36,49)), (3,(50,58)), (1,(60,72)), (4,(72,82)), (2,(94,102)), (5,(101,112)), (3,(112,120))]
        }

# 1 = with experience with binaural synthesis, 0 = without 
plot_colors = {1: "blue", 0: "red"}
plt.rcParams["figure.figsize"] = (15, 10)

# Little x offset due to difference between calibration and mid point
x_offset = 0.8

current_time = 0
for i in range(25):
    if(i % 2 == 0):
        time_map[1].append(current_time)
        time_map[2].append(current_time)
    
    if i % 2 == 0:
        current_time += 10
    else:
        current_time += 3

def handle_fig(fig, figure_fn):
    """Helper for figure saving and closing"""
    fig.tight_layout(pad=2, w_pad=0.5, h_pad=1.0)
    if save_figs:
        fig.savefig(os.path.join(plot_dir, figure_fn), dpi=figure_dpi)
    if close_figs:
        plt.close(fig)    

def plot_traces_individually(scenario, plot_hull=True):    
    scenario_files = filter_list_by_scenario(tracking_files, scenario)
    nrows = ncols = 5
    fig, ax = create_subplots_for_traces(scenario, title="")
    
    areas = []
    ways = []
    
    for idx, fn in enumerate(scenario_files):
        df = load_df_from_fn(fn)
        id, _ = get_id_and_scenario_from_fn(fn)
        
        df["x"] = df["x"].interpolate(limit_direction="both")
        df["y"] = df["y"].interpolate(limit_direction="both")
        
        points = np.zeros((len(df["x"]),2))
        points[:,0] = df["x"]
        points[:,1] = df["y"]
                
        hull = ConvexHull(points)
        
        points_hull = points[hull.vertices,:]
        
        x_hull = points_hull[:,0]
        y_hull = points_hull[:,1]        
        
        row = (id-1) // nrows
        col = (id-1) % ncols
        
        ax[row,col].set_ylim([-2, 2])
        ax[row,col].set_xlim([-2, 2])
        
        ax[row,col].plot(df["x"], df["y"])
                
        dist = df.diff().fillna(0)
        dist["xy_dist"] = np.sqrt((dist["x"].rolling(5).mean())**2+(dist["y"].rolling(5).mean()**2))
        
        area_of_motion = calc_poly_area(x_hull, y_hull)
        way_of_motion = dist["xy_dist"].sum()
        
        areas.append(area_of_motion)
        ways.append(way_of_motion)
    
        ax[row,col].set_title(f"{id}, A={area_of_motion:.1f}m², L={way_of_motion:.1f}m")    
        
        if plot_hull:            
            ax[row,col].plot(x_hull, y_hull)             
            
    handle_fig(fig, f"trace_{scenario}_individual_{plot_hull}.png")
    
    fig_area = plt.figure()
    plt.hist(areas)
    plt.xlabel("Flächen in m²", fontsize=axis_label_font_size)
    plt.ylabel("Häufigkeit", fontsize=axis_label_font_size)
    
    handle_fig(fig_area, f"area_hist_{scenario}.png")    
    
    fig_way = plt.figure()
    plt.hist(ways)
    plt.xlabel("Weglängen in m", fontsize=axis_label_font_size)
    plt.ylabel("Häufigkeit", fontsize=axis_label_font_size)
    
    handle_fig(fig_way, f"way_hist_{scenario}.png")
    
        
def plot_polar_example():    
    # Just take the first file as an example
    rotation_file = filter_list_by_scenario(filter_list_by_id(tracking_files, 1), 1)[0]
    df = load_df_from_fn(rotation_file)
    
    df = prepare_angles_for_interp(df)
    df = df.interpolate(limit_direction="both")
    
    fig = plt.figure()
    
    plt.polar(df["yaw"]/360*2*np.pi, np.arange(0, 1, 1/len(df)))
    
    handle_fig(fig, f"polar_example.png")
    
def plot_angle_processing_example():
    rotation_file = filter_list_by_scenario(filter_list_by_id(tracking_files, 1), 1)[0]
    df = load_df_from_fn(rotation_file)
    
    fig, ax = plt.subplots(2, sharex=True)
    ax[0].plot(df["timestamp"], df["yaw"].interpolate(limit_direction="both"))
    
    df_prepared = prepare_angles_for_interp(df)
    df_prepared = df_prepared.interpolate(limit_direction="both")
    
    ax[1].plot(df_prepared["timestamp"], df["yaw"].interpolate(limit_direction="both"))
    
    formatter = mt.FuncFormatter(timeTicks)
    ax[0].xaxis.set_major_formatter(formatter)
    ax[1].xaxis.set_major_formatter(formatter)
    
    ax[0].set_title(f"Vor Vorverarbeitung", fontsize=subplot_title_font_size)
    ax[1].set_title(f"Nach Vorverarbeitung", fontsize=subplot_title_font_size)
    
    handle_fig(fig, f"angle_preprocessing.png")
    
def plot_time_series_with_error_bars(scenario, feature):
    scenario_files = filter_list_by_scenario(tracking_files, scenario)
    fig = plt.figure()
    datapoints = []
    shortest_series = 999999999
    
    # Length of time series varies between persons, so find shortest and only plot
    # until length of shortest
    for fn in scenario_files:
        df = load_df_from_fn(fn)
        
        id, _ = get_id_and_scenario_from_fn(fn)        
        
        ax = plt.gca()
        formatter = mt.FuncFormatter(timeTicks)
        ax.xaxis.set_major_formatter(formatter)
        
        df["yaw"][df["yaw"] <0] += 360
        df = prepare_angles_for_interp(df)
        
        df = df.interpolate(limit_direction="both")
        
        datapoints.append(list(df[feature]))
        shortest_series = min(shortest_series, len(df))
        
    data = np.zeros((len(scenario_files),shortest_series))
    
    for idx, datapoint in enumerate(datapoints):
        data[idx,:] = datapoint[:shortest_series]     
    
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    
    plt.plot(df["timestamp"].iloc[:shortest_series], mean)    
    
    plt.fill_between(pd.to_numeric(df["timestamp"].iloc[:shortest_series]), mean-std, mean+std, alpha=0.3)        
    
    if scenario in [1,2]:
        for vline in time_map[scenario]:
            plt.gca().axvline(x=vline*10**9, color="grey", alpha=.5)
                
    if scenario == 3:
        for vline in segment_map[scenario][1::2]:
            plt.gca().axvspan(vline[0]*10**9, vline[1]*10**9, facecolor='grey', edgecolor='none', alpha=.2)
            
    plt.xlabel("Zeit in Minuten und Sekunden", fontsize=axis_label_font_size)
    if feature in ["yaw", "pitch", "roll"]:
        plt.ylabel("Winkel in °", fontsize=axis_label_font_size)
    else:
        plt.ylabel("Position in Meter", fontsize=axis_label_font_size)
        
    #plt.xlim([int(df["timestamp"].iloc[0].microseconds*1000), int(df["timestamp"].iloc[shortest_series].microseconds*1000)])
    
    handle_fig(fig, f"timeseries_{scenario}_{feature}.png")
    
def plot_x_y_time_series(scenario):   
    nrows = ncols = 5
    if scenario == 3 or scenario == 4:
        nrows = 11
        ncols = 2
    fig_x, ax_x = create_subplots_for_traces(scenario, nrows=nrows, ncols=ncols, polar=False)
    fig_y, ax_y = create_subplots_for_traces(scenario, nrows=nrows, ncols=ncols, polar=False) 
  
    formatter = mt.FuncFormatter(timeTicks)
    
    for idx, fn in enumerate(filter_list_by_scenario(tracking_files, scenario)):
        df = load_df_from_fn(fn)
        
        df = prepare_angles_for_interp(df)
        df = df.interpolate(limit_direction="both")
        
        id, _ = get_id_and_scenario_from_fn(fn)
    
        if scenario in [3,4]:
            row = (id-1) % nrows
            col = (id-1) // nrows
        else:
            row = (id-1) // nrows
            col = (id-1) % nrows
        
        
        formatter = mt.FuncFormatter(timeTicks)
        ax_x[row,col].xaxis.set_major_formatter(formatter)
        ax_y[row,col].xaxis.set_major_formatter(formatter)
        
        ax_x[row,col].plot(df["timestamp"], df["x"])
        ax_y[row,col].plot(df["timestamp"], df["y"])
        
        ax_x[row,col].set_title(f"{id}", fontsize=subplot_title_font_size)
        ax_y[row,col].set_title(f"{id}", fontsize=subplot_title_font_size)        
        
        ax_x[row,col].set_ylim([-2, 2])
        ax_y[row,col].set_ylim([-2, 2])
        
        if scenario == 4:
            for ax, vline in itertools.product([ax_x, ax_y], segment_map[scenario]):
                
                ax[row,col].axvline(vline[1][0]*10**9, color='green', alpha=.2)
                ax[row,col].axvline(vline[1][1]*10**9, color='red', alpha=.2)
                
    handle_fig(fig_x, f"trace_{scenario}_x.png")
    handle_fig(fig_y, f"trace_{scenario}_y.png")
    
def plot_yaw_data(scenario):
    nrows = ncols = 5
    if scenario == 3:
        nrows = 11
        ncols = 2
        
    fig_yaw, ax_yaw = create_subplots_for_traces(scenario, nrows=nrows, ncols=ncols, polar=False)#, title=f"Gier Positionen für Szenario {scenario}")    
    fig_speed, ax_speed = create_subplots_for_traces(scenario, nrows=nrows, ncols=ncols, polar=False)#, title=f"Gier Geschwindigkeiten für Szenario {scenario}")    
    fig_adf, ax_adf = create_subplots_for_traces(scenario, nrows=nrows, ncols=ncols, polar=False)#, title=f"ADF Ergebnisse")
    
    for idx, fn in enumerate(filter_list_by_scenario(tracking_files, scenario)):
        df = load_df_from_fn(fn)
        df = prepare_angles_for_interp(df)
        df = df.interpolate(limit_direction="both")
        
        id, _ = get_id_and_scenario_from_fn(fn)
        
        if not scenario == 3:
            row = (id-1) // nrows
            col = (id-1) % nrows       
        else:
            row = (id-1) // ncols
            col = (id-1) % ncols
                
        dist = df.diff().fillna(0)
        dist["yaw_speed"] = dist["yaw"].rolling(5).mean()/dist["timestamp"].dt.total_seconds()
        dist["xy_speed"] = np.sqrt((dist["x"].rolling(5).mean())**2+(dist["y"].rolling(5).mean()**2))/dist["timestamp"].dt.total_seconds()        
        
        plot_color = plot_colors[int(metadata[metadata["ID"] == id]["ErfBinSyn"])]
        
        # Plot this exact plot with modulo 360, because other plots will will be
        # displayed too small otherwise
        if scenario == 5 and int(id) == 6:
            df["yaw"] %= 360
            
        ax_yaw[row,col].plot(df["timestamp"], df["yaw"], color=plot_color)
        #ax_yaw[row,col].text(0.8,0.8, f"mean={df['yaw'].mean()}\nmax={df['yaw'].max()}\nsmoothness={stattools.acf(df['yaw'])[1]}", horizontalalignment='right', verticalalignment='top',transform=ax_yaw[row,col].transAxes)
        ax_speed[row,col].plot(df["timestamp"], dist["yaw_speed"], color=plot_color)
        axes = [ax_yaw, ax_speed, ax_adf]
#        df.set_index("timestamp", inplace=True)
        
        """ADF"""
        result = []
        for i in range(0, len(df), 75):
            segment = dist.iloc[i:i+150]["yaw_speed"].dropna()
            result.append(segment.std())        
            
        ax_adf[row,col].plot(np.linspace(0, df["timestamp"].iloc[-1].total_seconds()*10**9, len(result)), result )
        
        # Std Threshold for einschwingzeit
        threshold = 5
        ax_adf[row,col].axhline(y=threshold, color="red")    

        formatter = mt.FuncFormatter(timeTicks)
        
        for ax in axes:
            plt.setp(ax, xticks=np.arange(0, 154*10**9, 40*10**9))
            ax[row,col].set_title(f"{id}", fontsize=subplot_title_font_size)
            ax[row,col].xaxis.set_major_formatter(formatter)
             
        if scenario == 4:
            ax_adf[row,col].set_ylim([0, 10])
            
        speed_mean = np.abs(dist["yaw_speed"]).mean()
        speed_max = np.abs(dist["yaw_speed"]).max()
            
        omega_mean_txt = r"$\omega_{mean}$"
        omega_max_txt = r"$\omega_{max}$"
        
        ax_speed[row,col].set_title(f"{id}, {omega_mean_txt}={speed_mean:.1f}°/s, {omega_max_txt}={speed_max:.1f}°/s", fontsize=speed_subplot_title)
#       
            

        # Add lines from time or segment maps for for segmentation
        if scenario in [1,2]:
            for ax, vline in itertools.product(axes, time_map[scenario]):            
                ax[row,col].axvline(x=vline*10**9, color="grey", alpha=.5)
            
        if scenario == 3:
            for ax, vline in itertools.product(axes, segment_map[scenario][1::2]):
                
                ax[row,col].axvspan(vline[0]*10**9, vline[1]*10**9, facecolor='grey', edgecolor='none', alpha=.2)
                
        if scenario == 4:
            for ax, vline in itertools.product(axes, segment_map[scenario]):
                
                ax[row,col].axvline(vline[1][0]*10**9, color='green', alpha=.2)
                ax[row,col].axvline(vline[1][1]*10**9, color='red', alpha=.2)
                
            
    fig_yaw.text(0.5, 0.00, 'Zeit in Minuten und Sekunden', ha='center', va='bottom', fontsize=axis_label_font_size)
    fig_speed.text(0.5, 0.00, 'Zeit in Minuten und Sekunden', ha='center', va='bottom', fontsize=axis_label_font_size)
    fig_yaw.text(0.00, 0.5, "Gierwinkel in °", ha="left", va="center", rotation="vertical", fontsize=axis_label_font_size)
    fig_speed.text(0.00, 0.5, 'Gierwinkelgeschwindigkeit in °/sek', ha='left', va='center', rotation='vertical', fontsize=axis_label_font_size)
    fig_adf.text(0.5, 0.00, 'Zeit in Minuten und Sekunden', ha='center', va='bottom', fontsize=axis_label_font_size)   
    fig_adf.text(0.00, 0.5, "Standardabweichung Gierwinkel in °", ha="left", va="center", rotation="vertical", fontsize=axis_label_font_size)
    
    handle_fig(fig_yaw, f"yaw_{scenario}_position.png")
    handle_fig(fig_speed, f"yaw_{scenario}_speed.png")
    handle_fig(fig_adf, f"yaw_{scenario}_speed_std.png")
    
def plot_einschwingzeiten_yaw(scenario):
    results = {}
    
    for idx, fn in enumerate(filter_list_by_scenario(tracking_files, scenario)):
        df = load_df_from_fn(fn)
        df = prepare_angles_for_interp(df)
        df = df.interpolate(limit_direction="both")
        
        id, _ = get_id_and_scenario_from_fn(fn)
                
        dist = df.diff().fillna(0)
        dist["yaw_speed"] = dist["yaw"].rolling(5).mean()/dist["timestamp"].dt.total_seconds()
        dist["yaw_acc"] = dist["yaw_speed"].rolling(5).mean()/dist["timestamp"].dt.total_seconds()
        
        df.set_index("timestamp", inplace=True)
        
        """ADF"""
        result = []
        for i in range(0, len(df), 75):
            segment = dist.iloc[i:i+150]["yaw_speed"].dropna()
#            result.append(stattools.adfuller(segment)[1])
            result.append(segment.std())        
            
        results[id] = result
        
    segment_length = 13*2
    std_threshold = 5
    
    einschwingzeiten = defaultdict(list)
    
    for id, stds in results.items():
        for segment_idx, segment_start in enumerate(range(0, segment_length*12, segment_length)):
            segment = stds[segment_start:segment_length*(segment_idx+1)]    
            has_moved = False
            found_below_threshold = False
            for idx, item in enumerate(segment):
                if item > std_threshold:
                    has_moved = True
                if has_moved and item < std_threshold:
                    einschwingzeiten[id].append(idx)
                    found_below_threshold = True
                    break
            if not has_moved:
                einschwingzeiten[id].append(0)
            elif not found_below_threshold:
                einschwingzeiten[id].append(12)
            
                    
    einschwingzeiten_sec = {id: [t/2 for t in ts] for id, ts in einschwingzeiten.items()}
                
    fig_scatter = plt.figure()
    for key, val in einschwingzeiten_sec.items():
        if int(metadata[metadata["ID"] == key]["ErfBinSyn"]):
            plt.scatter(np.arange(1, len(val)+1), val, color="blue", alpha=0.2)
        else:
            plt.scatter(np.arange(1, len(val)+1), val, color="red", alpha=0.2)
            
    plt.xticks(np.arange(1,13,1))
    plt.xlabel("Segment", fontsize=axis_label_font_size)
    plt.ylabel("Einschwingzeit in Sekunden", fontsize=axis_label_font_size)
    
    handle_fig(fig_scatter, f"einschwingzeiten_{scenario}_yaw.png")
    
    fig_verteilung = plt.figure()
    plt.xticks(np.arange(1,13,1))
    plt.xlabel("Segment", fontsize=axis_label_font_size)
    plt.ylabel("Einschwingzeit in Sekunden", fontsize=axis_label_font_size)
    
    means = []
    errs = []
    
    channel_loc_data = defaultdict(list)
    
    if scenario not in [1,2,4]:
        return 
    
    for segment_idx in range(0,len(segment_map[scenario])):
        data = []
        for key, val in einschwingzeiten_sec.items():
            try:
                data.append(val[segment_idx])
            except IndexError:
                pass
        err = np.std(data)
        mean = np.mean(data)
        errs.append(err)
        means.append(mean)
        if scenario in [1,2]:
            channel_loc_data[segment_map[scenario][segment_idx]].extend(data)
        elif scenario in [4]:
            channel_loc_data[segment_map[scenario][segment_idx][0]].extend(data)
        plt.violinplot(data, [segment_idx+1], showmeans=True)
        
    handle_fig(fig_verteilung, f"einschwingzeiten_{scenario}_yaw_verteilung.png")
    
    fig_positionen = plt.figure()
    plt.xticks(np.arange(1,7,1))
    plt.xlabel("Quellenposition", fontsize=axis_label_font_size)
    plt.ylabel("Einschwingzeit in Sekunden", fontsize=axis_label_font_size)
    
    for channel_loc, data in channel_loc_data.items():
        err = np.std(data)
        mean = np.mean(data)
        plt.violinplot(data, [channel_loc], showmeans=True)
        
    handle_fig(fig_positionen, f"einschwingzeiten_{scenario}_yaw_positionen.png")
    
def plot_xy_data(scenario):
    nrows = ncols = 5
    fig_speed, ax_speed = create_subplots_for_traces(scenario, polar=False)#, title=f"Gier Positionen für Szenario {scenario}")   
    fig_speed_std, ax_speed_std = create_subplots_for_traces(scenario, polar=False)#, title=f"Gier Positionen für Szenario {scenario}")   
    
    total_df = pd.DataFrame()
    results = {}
        
    for idx, fn in enumerate(filter_list_by_scenario(tracking_files, scenario)):
        df = load_df_from_fn(fn)
        df = prepare_angles_for_interp(df)
        df = df.interpolate(limit_direction="both")
        
        id, _ = get_id_and_scenario_from_fn(fn)
        
        row = (id-1) // nrows
        col = (id-1) % ncols
                
        dist = df.diff().fillna(0)
        dist["yaw_speed"] = dist["yaw"].rolling(5).mean()/dist["timestamp"].dt.total_seconds()
        dist["yaw_acc"] = dist["yaw_speed"].rolling(5).mean()/dist["timestamp"].dt.total_seconds()
        dist["xy_speed"] = np.sqrt((dist["x"].rolling(5).mean())**2+(dist["y"].rolling(5).mean()**2))/dist["timestamp"].dt.total_seconds()
    
        ax_speed[row,col].plot(df["timestamp"], dist["xy_speed"])        
        
        formatter = mt.FuncFormatter(timeTicks)
        
        plt.setp(ax_speed, xticks=np.arange(0, 154*10**9, 40*10**9))
        
        speed_mean = dist["xy_speed"].mean()
        speed_max = dist["xy_speed"].max()
        
        v_mean_txt = r"$v_{mean}$"
        v_max_txt = r"$v_{max}$"
        
        ax_speed[row,col].set_title(f"{id}, {v_mean_txt}={speed_mean:.1f}m/s, {v_max_txt}={speed_max:.1f}m/s", fontsize=speed_subplot_title)
#        ax_speed[row,col].set_ylim([0, 0.4])
        ax_speed[row,col].xaxis.set_major_formatter(formatter)
        
        result = []
        for i in range(0, len(df), 75):
            segment = dist.iloc[i:i+150]["xy_speed"].dropna()
#            result.append(stattools.adfuller(segment)[1])
            result.append(segment.std())    
            
        ax_speed_std[row,col].plot(df["timestamp"][::75], result )
        ax_speed_std[row,col].axhline(y=0.05, color="red")
        
        for ax, vline in itertools.product([ax_speed], time_map[scenario]):
            ax[row,col].axvline(x=vline*10**9, color="grey", alpha=.5)
            
            ax_speed_std[row,col].axvline(x=vline*10**9, color="grey", alpha=.5)
#            plt.setp(ax_speed_std, xticks=np.arange(0, 154*10**9, 40*10**9))
            ax_speed_std[row,col].set_title(f"{id}", fontsize=subplot_title_font_size)
    #        ax_speed[row,col].set_ylim([0, 0.4])
            ax_speed_std[row,col].xaxis.set_major_formatter(formatter)   
            
        axes = [ax_speed, ax_speed_std]
        
        if scenario == 4:
            for ax, vline in itertools.product(axes, segment_map[scenario]):
                
                ax[row,col].axvline(vline[1][0]*10**9, color='green', alpha=.2)
                ax[row,col].axvline(vline[1][1]*10**9, color='red', alpha=.2)
            
        results[id] = result
        
    segment_length = 13*2
    std_threshold = 0.05
    
    einschwingzeiten = defaultdict(list)
    
    for id, stds in results.items():
        for segment_idx, segment_start in enumerate(range(0, segment_length*12, segment_length)):
            segment = stds[segment_start:segment_length*(segment_idx+1)]    
            has_moved = False
            found_below_threshold = False
            for idx, item in enumerate(segment):
                if item > std_threshold:
                    has_moved = True
                if has_moved and item < std_threshold:
                    einschwingzeiten[id].append(idx)
                    found_below_threshold = True
                    break
            if not has_moved:
                einschwingzeiten[id].append(0)
            elif not found_below_threshold:
                einschwingzeiten[id].append(12)
                    
    einschwingzeiten_sec = {id: [t/2 for t in ts] for id, ts in einschwingzeiten.items()}
            
    fig_speed.text(0.5, 0.00, 'Zeit in Minuten und Sekunden', ha='center', va='bottom', fontsize=axis_label_font_size)
    fig_speed.text(0.00, 0.5, "Geschwindigkeit in m/s", ha="left", va="center", rotation="vertical", fontsize=axis_label_font_size)
    fig_speed_std.text(0.5, 0.00, 'Zeit in Minuten und Sekunden', ha='center', va='bottom', fontsize=axis_label_font_size)
    fig_speed_std.text(0.00, 0.5, "Geschwindigkeit in m/s", ha="left", va="center", rotation="vertical", fontsize=axis_label_font_size)
                
    handle_fig(fig_speed, f"xy_{scenario}_speed.png")
    handle_fig(fig_speed_std, f"xy_{scenario}_speed_std.png")
    
    fig_scatter = plt.figure()
    for key, val in einschwingzeiten_sec.items():
        if int(metadata[metadata["ID"] == key]["ErfBinSyn"]):
            plt.scatter(np.arange(1, len(val)+1), val, color="blue", alpha=0.2)
        else:
            plt.scatter(np.arange(1, len(val)+1), val, color="red", alpha=0.2)
    plt.xticks(np.arange(1,13,1))
    plt.xlabel("Segment", fontsize=axis_label_font_size)
    plt.ylabel("Einschwingzeit in Sekunden", fontsize=axis_label_font_size)
    
    handle_fig(fig_scatter, f"einschwingzeiten_{scenario}_xy.png")
    
    fig_verteilung = plt.figure()
    plt.xticks(np.arange(1,13,1))
    plt.xlabel("Segment", fontsize=axis_label_font_size)
    plt.ylabel("Einschwingzeit in Sekunden", fontsize=axis_label_font_size)
    
    
    means = []
    errs = []
    
    channel_loc_data = defaultdict(list)
    
    if scenario not in [1,2,4]:
        return 
    
    for segment_idx in range(0,len(segment_map[scenario])):
        data = []
        for key, val in einschwingzeiten_sec.items():
            try:
                data.append(val[segment_idx])
            except IndexError:
                pass
        err = np.std(data)
        mean = np.mean(data)
        errs.append(err)
        means.append(mean)
        if scenario in [1,2]:
            channel_loc_data[segment_map[scenario][segment_idx]].extend(data)
        elif scenario in [4]:
            channel_loc_data[segment_map[scenario][segment_idx][0]].extend(data)        
        plt.violinplot(data, [segment_idx+1], showmeans=True)
    
    handle_fig(fig_verteilung, f"einschwingzeiten_{scenario}_xy_verteilung.png")
    
    fig_positionen = plt.figure()
    plt.xticks(np.arange(1,7,1))
    plt.xlabel("Quellenposition", fontsize=axis_label_font_size)
    plt.ylabel("Einschwingzeit in Sekunden", fontsize=axis_label_font_size)
    
    for channel_loc, data in channel_loc_data.items():
        err = np.std(data)
        mean = np.mean(data)
        plt.violinplot(data, [channel_loc], showmeans=True)
        
    handle_fig(fig_positionen, f"einschwingzeiten_{scenario}_xy_positionen.png")
    
    
def create_all_plots():
    plot_polar_example()
    plot_angle_processing_example()
    
    for scenario in range(1,6):
        plot_traces_individually(scenario)
        plot_time_series_with_error_bars(scenario, "pitch")
        plot_time_series_with_error_bars(scenario, "x")
        plot_time_series_with_error_bars(scenario, "y")
        plot_yaw_data(scenario)    
        plot_xy_data(scenario)
        
    for scenario in [1,2,4]:
        plot_einschwingzeiten_yaw(scenario)
        
    for scenario in [2,4,5]:
        plot_x_y_time_series(scenario)    
    
if __name__ == "__main__":
    create_all_plots()
    
