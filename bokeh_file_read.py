#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  9 22:06:13 2020

@author: lukishyadav
"""

from bokeh.io import curdoc
from bokeh.models.widgets import FileInput
from pybase64 import b64decode
import pandas as pd
import io
from bokeh.models.widgets import Button, RadioButtonGroup, Select, Slider,TextInput,TextAreaInput
from bokeh.layouts import column,layout,row,widgetbox

from bokeh.plotting import figure, output_file, show

def upload_fit_data(attr, old, new):
    print("fit data upload succeeded")

    decoded = b64decode(new)
    f = io.BytesIO(decoded)
    global df
    print(type(new))
    df = pd.read_csv(f)
    #print(new_df)

file_input = FileInput(accept=".csv,.json,.txt,.pdf,.xls")
file_input.on_change('value', upload_fit_data)


p = figure(
#    x_range=(minlng,maxlng),
#    y_range=(minlat, maxlat),
    #x_axis_type=map_repr,
    #y_axis_type=map_repr,
    #title='IDLE Vehicles Map',
    match_aspect=True,
    tools="pan,wheel_zoom,box_zoom,tap,box_select,reset,save"
    #tools='tap'
)


p.circle(x=[], y=[], 
                  #size=cluster_point_size,
                  fill_alpha=0.2,
                  color="royalblue",line_alpha=0
                  #line_color='black'
                  )

def update_click():
    global df
    p.circle(x=df['end_long'], y=df['end_lat'], fill_alpha=0.2,color="royalblue",line_alpha=0)
    
    #p.circle(x=[], y=[], fill_alpha=0.2,color="royalblue",line_alpha=0)
    
bt = Button(label='Update Plot')
bt.on_click(update_click) 



    

doc=curdoc()
doc.add_root(layout(row(file_input,p,bt)))