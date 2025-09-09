import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import smplotlib

from tqdm import tqdm


def plot_the_corner(sliders):
  # Names for each variable (length N)
  colnames    = ["t_infall",           "pericenter_radius",   "apocenter_radius", "angle_orbit_disk_infall", "sigma_stellar",       "r_50_stars",             "stellar_mass",             "dm_mass"]
  constraints = [(0, 13.82),           (-10, 200),            (-10, 400),         (-5, 95),                  (2, 150),              (0.011, 3),              (4E3, 2E9),                  (1E8, 5E10)]
  lims        = [(0, 13.82),           (-10, 200),            (-10, 400),         (-5, 95),                  (2, 150),              (0.011, 3),              (4E3, 2E9),                  (1E8, 5E10)]
  scales      = ["linear",             "linear",              "linear",           "linear",                  "log",                 "log",                    "log",                      "log"]
  labels      = [r"$t_{infall} [Gyr]$", r"$r_{peri}$ [kpc]", r"$r_{apo}$ [kpc]", r"$\alpha_{infall}$ [deg]", r"$\sigma_*$ [km/s]", r"Stellar $r_{h}$ [kpc]", r"$M_{stars}$ [$M_\odot$]", r"$M_{DM}$ [$M_\odot$]"]
  N = len(labels)
    
  constraints = [list(slider.value) for slider in sliders]
    
  # Create N×N subplots
  fig, axes = plt.subplots(
      N, N,
      figsize=(3.3*(N), 3.3*(N)),
      sharex=False,   
      sharey=False,
      constrained_layout=True
  )
  
  
  
  # Loop over the grid
  for i in range(1, N):
      for j in range(N-1):
          ax = axes[i, j]
          if i != j:
              ax.set_xscale(scales[j])
              ax.set_yscale(scales[i])
              ax.set_xlim(lims[j])
              ax.set_ylim(lims[i])
  
  
              if colnames[i] == "apocenter_radius" and colnames[j] == "pericenter_radius":
                  ax.plot(np.linspace(0,300,100), np.linspace(0,300,100), lw=1.2, color="gray", ls="--")
              
              if i == N-1:
                  ax.set_xlabel(labels[j], fontsize=22)
              else:
                  ax.set_xticklabels([])
                  
              if j == 0:
                  ax.set_ylabel(labels[i], fontsize=22)
              else:
                  ax.set_yticklabels([])
                  
  
  
  for i in range(N):
      axes[i,i].set_xlim(lims[i])
      axes[i,i].set_xscale(scales[i])
      #axes[i, i].set_yticklabels([])
      #axes[i, i].set_xticklabels([])
  
      axes[i, i].xaxis.labelpad = 11
      axes[i, i].yaxis.labelpad = 10
      axes[i, i].xaxis.set_label_position("top")
      axes[i, i].yaxis.set_label_position("right")
  
      axes[i, i].set_xlabel(labels[i], fontsize=22)
      axes[i, i].set_ylabel("Counts", fontsize=22)
  
      axes[i, i].tick_params(axis='x', top=True, labeltop=True, bottom=False, labelbottom=False, pad=0.02)
      axes[i, i].tick_params(axis='y', left=False, labelleft=False, right=True, labelright=True, pad=5)
  
      for j in range(N):
          if i<j:
              axes[i,j].set_axis_off()
  
  
  
  k = 0
  pd_full = pd.DataFrame()
  mk = ["p", "D", "v", "^", "o", "s", "*", "X"]
  for h in tqdm(['599', '666', '715', '685', '685x09', '685x095', '685x11', '685x12']):
      df_load = pd.read_csv(f"./vintergatan_streams_catalogue/halo{h}_streams_catalogue.csv") #
      df = df_load[(~df_load["stellar_mass"].isna()) & (df_load["apocenter_radius"] > 0)]  
  
      masks = []
      for con, col in zip(constraints, colnames):
          masks.append((df[col] > con[0]) & (con[1] > df[col]))
          
      mask = masks[0]
      for m in masks[1:]:
          mask = mask & m
  
      df = df[mask]
  
      
      pd_full = pd.concat([pd_full, df])
  
      
      df_phase  = df[df["phase-state_zcurr"] == 2] 
      df_stream = df[df["phase-state_zcurr"] == 1] 
      df_intact = df[df["phase-state_zcurr"] == 0] 
  
      for i in range(1, N):
          for j in range(N-1):
              if i<=j:
                  continue
  
  
              axes[i, j].scatter(df_intact[colnames[j]], df_intact[colnames[i]], marker=mk[k], color="green", s=10, alpha=0.7, zorder=9)
              axes[i, j].scatter(df_stream[colnames[j]], df_stream[colnames[i]], marker=mk[k], color="red", s=10, alpha=0.7, zorder=8)
              axes[i, j].scatter(df_phase[colnames[j]], df_phase[colnames[i]], marker=mk[k], color="blue", s=10, alpha=0.5, zorder=7)
  
      k += 1
  
  
  for i in range(N):
      ax = axes[i, i]
      # pick your global min/max from lims
      lo, hi = lims[i]
      bins = 6
  
      # construct edges
      if scales[i] == "log":
          # avoid zero or negative
          lo_, hi_ = max(lo, 1e-8), hi
          edges = np.logspace(np.log10(lo_), np.log10(hi_), bins + 1)
      else:
          edges = np.linspace(lo, hi, bins + 1)
          
      edges = np.concatenate((edges, [edges[-1] + (edges[-1] - edges[-2])]))
  
      # collect data for each state
      data_intact = pd_full.loc[pd_full["phase-state_zcurr"]==0, colnames[i]].values
      data_stream = pd_full.loc[pd_full["phase-state_zcurr"]==1, colnames[i]].values
      data_phase  = pd_full.loc[pd_full["phase-state_zcurr"]==2, colnames[i]].values
  
      # compute histograms
      h_int, _ = np.histogram(data_intact, bins=edges)
      h_str, _ = np.histogram(data_stream, bins=edges)
      h_phs, _ = np.histogram(data_phase,  bins=edges)
  
      # plot as stepped lines
      ax.step(edges[:-1], h_int, where='post', color='green',  lw=1.5, label='Intact')
      ax.step(edges[:-1], h_str, where='post', color='red',    lw=1.5, label='Stream')
      ax.step(edges[:-1], h_phs, where='post', color='blue',   lw=1.5, label='Phase Mixed')
  
  
  
  
  
  
      
  
  
      #axes[i, i].text(0.5, 0.9, labels[i], va="top", ha="center")
  
  
      
      
  from matplotlib.pyplot import Line2D
  legend_handles = [
      Line2D([0],[0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Intact'),
      Line2D([0],[0], marker='o', color='w', markerfacecolor='red',   markersize=10, label='Streams'),
      Line2D([0],[0], marker='o', color='w', markerfacecolor='blue',  markersize=10, label='Phase Mixed'),
  ]
  fig.legend(handles=legend_handles, title='Dynamical State', loc='upper right', frameon=True, fontsize="large")
  
  plt.tight_layout()
  fig.subplots_adjust(hspace=0.075, wspace=0.075)
  return fig
