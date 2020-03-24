#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 18:38:01 2020

@author: dazmer
"""

import plotly.graph_objects as go
import numpy as np

import csv

res_vals = {}
wsz_l = []
asz_l = []
ssz_l = []
with open('registering_uncompleted_canny.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
        ssz, wsz, asz, msen, pepn, time = row[0].split(",")
        # if msen == "null":
        wsz_l.append(wsz)
        asz_l.append(asz)
        ssz_l.append(ssz)
        if wsz not in res_vals: res_vals[wsz] = {}
        if asz not in res_vals[wsz]: res_vals[wsz][asz] = {}
        if(msen != "null"):
            res_vals[wsz][asz][ssz] = float(pepn)
        else:
            res_vals[wsz][asz][ssz] = np.nan
    wsz_labels = np.unique(wsz_l)
    asz_labels = np.unique(asz_l)
    ssz_labels = np.unique(ssz_l)
    
    surfaces = []
    for t, lab_ssz in enumerate(ssz_labels):
        res_grid = np.array(np.zeros((len(wsz_labels),len(asz_labels))))
        res_grid[:] = np.nan
        for i, lab_wsz in enumerate(wsz_labels):
            for j, lab_asz in enumerate(asz_labels):
                if(lab_wsz in res_vals):
                    if(lab_asz in res_vals[lab_wsz]):
                        if(lab_ssz in res_vals[lab_wsz][lab_asz]):
                            res_grid[i][j] = res_vals[lab_wsz][lab_asz][lab_ssz]
                        # else:
                            # res_grid[i][j] = 
        if(t==0):
            surface = go.Surface(z=res_grid)
        else:
            surface = go.Surface(z=res_grid, showscale=False, opacity=0.9)
        surfaces.append(surface)
        print(surface)
        fig = go.Figure()
        fig.add_trace(surface)
        fig.update_layout(
            width=800,
            height=900,
            autosize=False,
            margin=dict(t=0, b=0, l=0, r=0),
            template="plotly_white",
        )

        fig.update_scenes(
            aspectratio=dict(x=1, y=1, z=0.7),
            aspectmode="manual"
        )
        
        fig.update_layout(
            updatemenus=[
                dict(
                    type = "buttons",
                    direction = "left",
                    buttons=list([
                        dict(
                            args=["type", "surface"],
                            label="3D Surface",
                            method="restyle"
                        ),
                        dict(
                            args=["type", "heatmap"],
                            label="Heatmap",
                            method="restyle"
                        )
                    ]),
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.11,
                    xanchor="left",
                    y=1.1,
                    yanchor="top"
                ),
                dict(
                    type = "buttons",
                    direction = "left",
                    buttons=list([
                        dict(
                            args=["type", "surface"],
                            label="3D Surface",
                            method="restyle"
                        ),
                        dict(
                            args=["type", "heatmap"],
                            label="Heatmap",
                            method="restyle"
                        )
                    ]),
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.11,
                    xanchor="left",
                    y=1.1,
                    yanchor="top"
                )
            ]
        )
        
        fig.show(renderer="browser")
        # print(', '.join(row))

    