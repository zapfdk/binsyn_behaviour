#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 14:07:39 2019

@author: dominik
"""
import os
import time

from scipy.io import wavfile
import matplotlib.pyplot as plt
import matplotlib.ticker as mt
import numpy as np

audio_dir = "./audio/final"

fs = 48000
downsampling_factor = 100

from visualization import handle_fig

def timeTicks(x, pos):    
    seconds = x / fs * downsampling_factor
    minutes, seconds = divmod(seconds, 60)
    return f"{int(minutes):02}:{int(seconds):02}"

def plot_wavefiles(wavefile_names):
    for name, fn in wavefile_names.items():
        filepath = os.path.join(audio_dir, fn)
        fs, data = wavfile.read(filepath)
        data = data.astype(np.float16)[::downsampling_factor]
        fs /= downsampling_factor
        
        channels = data.shape[1]
        
        fig, ax = plt.subplots(channels, sharex=True, sharey=True)        
        fig.subplots_adjust(hspace=0)
        formatter = mt.FuncFormatter(timeTicks)
        plt.gca().xaxis.set_major_formatter(formatter)
        
        for i in range(channels):
            ax[i].plot(data[:,i])
            ax[i].set_ylabel(f"{i+1}")
            ax[i].set_xlim([0, len(data)])
            
            if name in ["Szenario_1", "Szenario_2"]:            
                start = 0
                audio_length = 10
                pause_length = 1
                while start < data.shape[0]:
                    ax[i].axvspan(start, start+fs*audio_length, facecolor='green', edgecolor='none', alpha=.2)
                    start += fs*audio_length
                    ax[i].axvspan(start, start+fs*pause_length, facecolor='black', edgecolor='none', alpha=.2)
                    start += fs*pause_length
                    ax[i].axvspan(start, start+fs*pause_length, facecolor='red', edgecolor='none', alpha=.2)
                    start += fs*pause_length
                    ax[i].axvspan(start, start+fs*pause_length, facecolor='black', edgecolor='none', alpha=.2)
                    start += fs*pause_length              
                
        handle_fig(fig, f"waveplot_{name}.png")
        
        
if __name__ == "__main__":
    wavefile_names = {"scenario_1": "scenario_1.wav",
                      "scenario_2": "scenario_2.wav",
                      "scenario_3": "scenario_3.wav",
                      "scenario_4": "scenario_4.wav",
                      "scenario_5": "Demo-Frank2-rev.wav",
                      "training": "DerPanther.wav"}

    plot_wavefiles(wavefile_names)
    
