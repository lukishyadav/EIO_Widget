#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 13:44:54 2020

@author: lukishyadav
"""

import time

mstart=time.time()

from bokeh.io import curdoc
#import logging
from bokeh.layouts import column,layout,row,widgetbox
import pandas as pd
#import my_module
import datetime
import seaborn as sns
from pyproj import Proj
from bokeh.plotting import figure, show, output_file, ColumnDataSource
from bokeh.transform import factor_cmap
from bokeh.tile_providers import CARTODBPOSITRON,OSM,ESRI_IMAGERY
import numpy as np
from sklearn.cluster import DBSCAN 
from bokeh.models.widgets import Button, RadioButtonGroup, Select, Slider,TextInput,TextAreaInput
from bokeh.models import TextInput,LinearColorMapper
from collections import Counter
from bokeh.util.hex import hexbin
from bokeh.transform import linear_cmap
from bokeh.models import RangeSlider

from datetime import date

from bokeh.models.widgets import DateRangeSlider,DateSlider
from h3 import h3
from geojson.feature import *
import json
import time
from bokeh.models import GeoJSONDataSource

from bokeh.models import Toggle
from bokeh.models import Div
from datetime import timedelta
from bokeh.models import Select
from bokeh.tile_providers import CARTODBPOSITRON, get_provider,OSM,ESRI_IMAGERY

from bokeh.models.renderers import  TileRenderer

import numpy as np

from bokeh.io import curdoc, show
from bokeh.models import ColumnDataSource, Grid, LinearAxis, Plot, Quadratic

from math import radians, cos, sin, asin, sqrt

N = 9
x = np.linspace(-2, 2, N)
y = x**2

source = ColumnDataSource(dict(
        x=x,
        y=y,
        xp02=x+0.4,
        xp01=x+0.1,
        yp01=y+0.2,
    )
)

plot = Plot(
    title=None, plot_width=300, plot_height=300,
    min_border=0, toolbar_location=None)

glyph = Quadratic(x0="x", y0="y", x1="xp02", y1="y", cx="xp01", cy="yp01", line_color="#4daf4a", line_width=3)
plot.add_glyph(source, glyph)

"""
xaxis = LinearAxis()
plot.add_layout(xaxis, 'below')

yaxis = LinearAxis()
plot.add_layout(yaxis, 'left')
"""

#plot.add_layout(Grid(dimension=0, ticker=xaxis.ticker))
#plot.add_layout(Grid(dimension=1, ticker=yaxis.ticker))

curdoc().add_root(plot)

show(plot)



import pandas as pd
infile='widget4_data/darwin_rental_data.csv'

df=pd.read_csv(infile)


def haversine(x):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [x['end_long'], x['end_lat'], x['start_long'], x['start_lat']])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r*1000

df['haversine_distance']=df.apply(haversine,axis=1)
df=df.head(10)

def convert_to_mercator(lngs, lats):
    projection = Proj(init='epsg:3857')
    xs = []
    ys = []
    for lng, lat in zip(lngs, lats):
        x, y = projection(lng, lat)
        xs.append(x)
        ys.append(y)
    return xs, ys



cl=['start_lat', 'start_long', 'end_lat','end_long']


for lcol in list(range(0,len(cl),2)):    
    df['mrc_'+cl[lcol+1]],df['mrc_'+cl[lcol]]=convert_to_mercator(df[cl[lcol+1]], df[cl[lcol]])
    
source = ColumnDataSource(dict(
        x=df['mrc_start_long'],
        y=df['mrc_start_lat'],
        x1=df['mrc_end_long'],
        y1=df['mrc_end_lat'],
        cx=(df['mrc_start_long']+df['mrc_end_long'])/2,
        cy=df['mrc_start_lat']+df['haversine_distance']/8,
    )
)

    
map_repr='mercator'
# set up/draw the map
p = figure(
#    x_range=(minlng,maxlng),
#    y_range=(minlat, maxlat),
    x_axis_type=map_repr,
    y_axis_type=map_repr,
    title='IDLE Vehicles Map',
    match_aspect=True
)

#ESRI_IMAGERY,OSM
tile_provider = get_provider(CARTODBPOSITRON)
p.add_tile(tile_provider)

#4daf4
#darkslategrey   
#cx="cx", cy="cy"
glyph = p.quadratic(x0="x", y0="y", x1="x1", y1="y1",cx='cx',cy='cy', line_color="darkslategrey", line_width=2,source=source)
#plot.add_glyph(source, glyph)

"""
xaxis = LinearAxis()
plot.add_layout(xaxis, 'below')

yaxis = LinearAxis()
plot.add_layout(yaxis, 'left')
"""

#plot.add_layout(Grid(dimension=0, ticker=xaxis.ticker))
#plot.add_layout(Grid(dimension=1, ticker=yaxis.ticker))

curdoc().add_root(p)

show(p)
    
    