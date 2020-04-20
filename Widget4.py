#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 10:02:40 2020

@author: lukishyadav
"""
import time

mstart=time.time()

from bokeh.io import curdoc
import logging
from bokeh.layouts import column,layout,row,widgetbox
import pandas as pd
#import my_module
import datetime
import seaborn as sns
from pyproj import Proj
from bokeh.plotting import figure, show, output_file, ColumnDataSource
from bokeh.transform import factor_cmap
from bokeh.tile_providers import CARTODBPOSITRON 
import numpy as np
from sklearn.cluster import DBSCAN 
from bokeh.models.widgets import Button, RadioButtonGroup, Select, Slider,TextInput,TextAreaInput
from bokeh.models import TextInput,LinearColorMapper
from collections import Counter
from bokeh.util.hex import hexbin
from bokeh.transform import linear_cmap

from datetime import date

from bokeh.models.widgets import DateRangeSlider,DateSlider
from h3 import h3
from geojson.feature import *
import json
import time
from bokeh.models import GeoJSONDataSource

from bokeh.models import Toggle


map_repr='mercator'

infile='generated_data/rentals_wave3.csv'


#PRESETS

max_res = 15

RESOLUTION=7

C_T=1


svalue="2020-01-03 00:00:00"
evalue="2020-01-03 01:00:00"



list_hex_edge_km = []
list_hex_edge_m = []
list_hex_perimeter_km = []
list_hex_perimeter_m = []
list_hex_area_sqkm = []
list_hex_area_sqm = []

for i in range(0,max_res + 1):
    ekm = h3.edge_length(resolution=i, unit='km')
    em = h3.edge_length(resolution=i, unit='m')
    list_hex_edge_km.append(round(ekm,3))
    list_hex_edge_m.append(round(em,3))
    list_hex_perimeter_km.append(round(6 * ekm,3))
    list_hex_perimeter_m.append(round(6 * em,3))
    
    akm = h3.hex_area(resolution=i, unit='km^2')
    am = h3.hex_area(resolution=i, unit='m^2')
    list_hex_area_sqkm.append(round(akm,3))
    list_hex_area_sqm.append(round(am,3))

    
df_meta = pd.DataFrame({"edge_length_km" : list_hex_edge_km,
                        "perimeter_km" : list_hex_perimeter_km,
                        "area_sqkm": list_hex_area_sqkm,
                        "edge_length_m" : list_hex_edge_m,
                        "perimeter_m" : list_hex_perimeter_m,
                        "area_sqm" : list_hex_area_sqm
                       })
                      
df_meta[["edge_length_km","perimeter_km","area_sqkm", "edge_length_m", "perimeter_m" ,"area_sqm"]]



df=pd.read_csv(infile)



#df=df[(df['rental_started_at']>svalue) & (df['rental_started_at']<evalue)]



#df=df[(df['rental_started_at']>svalue) & (df['rental_started_at']<evalue)]

df=df[(df['rental_started_at']>str(datetime.datetime(2020, 1, 3,1))) & (df['rental_started_at']<str(datetime.datetime(2020, 1, 3,2)))]
#df=df[(df['rental_ended_at_x']<=svalue) & (df['rental_started_at_y']>=evalue)]


#df=dd.read_csv(infile)

#display_columns=df.columns

def convert_to_mercator(lngs, lats):
    projection = Proj(init='epsg:3857')
    xs = []
    ys = []
    for lng, lat in zip(lngs, lats):
        x, y = projection(lng, lat)
        xs.append(x)
        ys.append(y)
    return xs, ys


latlong=['mrc_start_lat','mrc_start_long']

latlong=['start_lat','start_long']

def counts_by_hexagon(df, resolution):    
    '''Use h3.geo_to_h3 to index each data point into the spatial index of the specified resolution.
      Use h3.h3_to_geo_boundary to obtain the geometries of these hexagons'''

    #df = df[["latitude","longitude"]]
    df=df[latlong]
    
    #df["hex_id"] = df.apply(lambda row: h3.geo_to_h3(row["latitude"], row["longitude"], resolution), axis = 1)
    df["hex_id"] = df.apply(lambda row: h3.geo_to_h3(row[latlong[0]], row[latlong[1]], resolution), axis = 1)
    
    df_aggreg = df.groupby(by = "hex_id").size().reset_index()
    df_aggreg.columns = ["hex_id", "value"]
    

    df_aggreg["geometry"] =  df_aggreg.hex_id.apply(lambda x: 
                                                           {    "type" : "Polygon",
                                                                 "coordinates": 
                                                                [h3.h3_to_geo_boundary(h3_address=x,geo_json=True)]
                                                            }
                                                        )

    """
    df_aggreg["center"] =  df_aggreg.hex_id.apply(lambda x: 
                                                           {    "type" : "Polygon",
                                                                 "coordinates": 
                                                                [h3.h3_to_geo(h3_address=x)]
                                                            }
                                                        )
    """    
        
    return df_aggreg




def hexagons_dataframe_to_geojson(df_hex, file_output = None):
    
    '''Produce the GeoJSON for a dataframe that has a geometry column in geojson format already, along with the columns hex_id and value '''
    
    list_features = []
    
    for i,row in df_hex.iterrows():
        
        
        #Converting to Mercator
        v=np.array(row["geometry"]['coordinates'][0])
        v[:,0],v[:,1]=convert_to_mercator(v[:,0], v[:,1])
        row["geometry"]['coordinates'][0]=v.tolist()
            
        feature = Feature(geometry = row["geometry"] , id=row["hex_id"], properties = {"value" : row["value"]})
        list_features.append(feature)
        
    feat_collection = FeatureCollection(list_features)
    
    geojson_result = json.dumps(feat_collection)
    
    #optionally write to file
    if file_output is not None:
        with open(file_output,"w") as f:
            json.dump(feat_collection,f)
    
    return geojson_result





start=time.time()
df_aggreg= counts_by_hexagon(df = df, resolution = RESOLUTION)
print(time.time()-start)




geojson_data = hexagons_dataframe_to_geojson(df_hex = df_aggreg)

geo_source = GeoJSONDataSource(geojson=geojson_data)



"""
dictionary = dict(
    Q=purpose['Q'],
    R=purpose['R'],
    C=purpose['C'],
    counts=purpose['counts']    
    #label=datapoints_df['label'],
    #time=datapoints_df['start_datetime']
    )


dictionary={}

for col_name in purpose.columns:
# if col_name not in [X,Y]:
  dictionary[col_name]=purpose[col_name]
purpose_source.data = dictionary 
"""

"""

SLIDE HANDLER

"""

def my_slider_handler():
    #print(toggle_small.active)
    range_slider1=sdate_range_slider
    RESOLUTION=resolution_slider.value
    C_T=int(count_threshold.value)


    df=pd.read_csv(infile)
    if isinstance(range_slider1.value[0], (int, float)):
    # pandas expects nanoseconds since epoch
        start_date = pd.Timestamp(float(range_slider1.value[0])*1e6)
        end_date = pd.Timestamp(float(range_slider1.value[1])*1e6)
    else:
        start_date = pd.Timestamp(range_slider1.value[0])
        end_date = pd.Timestamp(range_slider1.value[1])
        
    print(start_date,end_date)
    
    ms=max(df['rental_started_at'])
    
    df=df[(df['rental_started_at']>str(start_date)) & (df['rental_started_at']<str(end_date))]
    
    start=time.time()
    df_aggreg= counts_by_hexagon(df = df, resolution = RESOLUTION)
    print(time.time()-start)
    
    geojson_data = hexagons_dataframe_to_geojson(df_hex = df_aggreg)
    
    if toggle_hexes.active==True:
        print("Hex remove Condition")
        #geo_source.geojson=json.dumps({"type":"FeatureCollection","features":[{"type":"Feature","id":"832830fffffffff","geometry":{"type":"Polygon","coordinates":[[[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]]]},"properties":{"value":0}}]})
        geo_source.geojson=json.dumps({"type":"FeatureCollection","features":[{"type":"Feature","id":"832830fffffffff","geometry":{"type":"Polygon","coordinates":[[[ms,ms] for x in range(7)]]},"properties":{"value":0}}]})
    else:  
        geo_source.geojson=geojson_data        


    
    #datapoints_source = ColumnDataSource(df)
    if toggle_small.active==True:
        dictionary={}
        for col_name in df.columns:
        # if col_name not in [X,Y]:
          dictionary[col_name]=[]
        datapoints_source.data = dictionary 
    else:    
        dictionary={}
        for col_name in df.columns:
        # if col_name not in [X,Y]:
          dictionary[col_name]=df[col_name]
        datapoints_source.data = dictionary 
    #th=list(set(purpose['C']))
    #th=[str(i) for i in th]
    #mapper['transform'].factors=th
    
    #AttributeError: unexpected attribute 'factors' to LinearColorMapper, possible attributes are high, high_color, js_event_callbacks, js_property_callbacks, low, low_color, name, nan_color, palette, subscribed_events or tags

    
    pass



"""
WIDGETS

"""

sdate_input = TextInput(value="2020-01-03 00:00:00", title="Start Date: (YYYY-MM-DD HH:MM:SS)")
#sdate_input.on_change("value", my_slider_handler)

edate_input = TextInput(value="2020-01-03 01:00:00", title="End Date: (YYYY-MM-DD HH:MM:SS)")
#edate_input.on_change("value", my_slider_handler)
count_threshold = TextInput(value="1", title="Count Threshold Value")

resolution_slider=Slider(start=1, end=15, value=7, step=1, title="H3 Resolutions")

sdate_range_slider = DateRangeSlider(title="Date Range: ", start=datetime.datetime(2020, 1, 1,1), end=datetime.datetime(2020, 3, 20,1), value=(datetime.datetime(2020, 1, 3,1), datetime.datetime(2020, 1, 3,2)),format="%x,%X")

#sdate_range_slider = DateRangeSlider(title="Date Range: ", start=datetime.datetime(2017, 1, 1,1), end=datetime.datetime(2017, 2, 7,2), value=(datetime.datetime(2017, 9, 7,1), datetime.datetime(2017, 9, 7,2)),format="%Y-%m-%d %H")
#sdate_range_slider = DateRangeSlider(title="Date Range: ", start=datetime.datetime(2017, 1, 1,1), end=datetime.datetime(2017, 2, 7,2), value=(datetime.datetime(2017, 9, 7,1), datetime.datetime(2017, 9, 7,2)),step=1)
#sdate_range_slider.on_change("value", my_slider_handler)
#sdate_range_slider = DateSlider(title="Date Range: ", start=datetime.datetime(2017, 1, 1,1), end=datetime.datetime(2019, 9, 7,2), value=(datetime.datetime(2017, 9, 7,1), datetime.datetime(2017, 9, 7,2)), step=1)

bt = Button(label='Update Plot')
bt.on_click(my_slider_handler)


def toggle_small_handler(attr, old, new):
    print(toggle_small.active)
    my_slider_handler()
    
def toggle_hexes_handler(attr, old, new):
    print(toggle_hexes.active)
    my_slider_handler()

toggle_small = Toggle(label="Toggle Points", button_type="success")
toggle_small.on_change("active",  toggle_small_handler)


toggle_hexes = Toggle(label="Toggle Hexes", button_type="success")
toggle_hexes.on_change("active",  toggle_hexes_handler)


from bokeh.tile_providers import CARTODBPOSITRON 

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

p.add_tile(CARTODBPOSITRON)




"""
HOVERTOOL

"""
display_columns1=df.columns
from bokeh.models import HoverTool
TOOLTIP1=HoverTool()
TOOLTIP_list1=['<b style="color:MediumSeaGreen;">'+name_cols+':'+'</b><b>'+' @{'+name_cols+'}</b>' for name_cols in display_columns1]
#TOOLTIP=[(name_cols,'@{'+name_cols+'}') for name_cols in display_columns]
TOOLTIP_end1 = "<br>".join(TOOLTIP_list1)

TOOLTIP1.tooltips= """
    <style>
        .bk-tooltip>div:not(:first-child) {display:none;}
    </style>"""+TOOLTIP_end1




display_columns2=['value']
from bokeh.models import HoverTool
TOOLTIP2=HoverTool()
TOOLTIP_list2=['<b style="color:MediumSeaGreen;">'+name_cols+':'+'</b><b>'+' @{'+name_cols+'}</b>' for name_cols in display_columns2]
#TOOLTIP=[(name_cols,'@{'+name_cols+'}') for name_cols in display_columns]
TOOLTIP_end2 = "<br>".join(TOOLTIP_list2)

TOOLTIP2.tooltips= """
    <style>
        .bk-tooltip>div:not(:first-child) {display:none;}
    </style>"""+TOOLTIP_end2
    

    
p.add_tools(TOOLTIP1)
p.add_tools(TOOLTIP2)
           



from bokeh.palettes import Oranges,OrRd,RdYlGn

#th=list(set(purpose['C']))
#th=[str(i) for i in th]

#mapper=factor_cmap('C', RdYlGn[5],th )
#mapper=linear_cmap('C', RdYlGn[5], 0, max(purpose.C))

color_mapper = LinearColorMapper(palette=RdYlGn[5])
p.patches('xs', 'ys', fill_alpha=0.7, fill_color={'field': 'value', 'transform': color_mapper},
          line_color='white', line_width=0.5, source=geo_source)

from bokeh.io import output_file, show 

#output_file("hex_tile.html")


#VERIFICATION STUFF
datapoints_source = ColumnDataSource(df)

"""
dictionary = dict(
    x=df['mrc_start_long'],
    y=df['mrc_start_lat'],
    #label=datapoints_df['label'],
    #time=datapoints_df['start_datetime']
    )

for col_name in display_columns1:
# if col_name not in [X,Y]:
  dictionary[col_name]=df[col_name]
  
datapoints_source.data = dictionary 
"""
p.circle(x='mrc_start_long', y='mrc_start_lat', 
                  #size=cluster_point_size,
                  fill_alpha=0.2,
                  source=datapoints_source,color="royalblue",line_alpha=0.4
                  #line_color='black'
                  )

#show(p)

     
layout = column(row(
            column(widgetbox(sdate_input),
                widgetbox(edate_input),
                widgetbox(count_threshold),   
                widgetbox(resolution_slider),
                widgetbox(toggle_small),
                widgetbox(toggle_hexes),
                widgetbox(bt)
                #widgetbox(hovertool_widget)
                ,width=400),   
                #widgetbox(slider,width=350),
                #widgetbox(Min_n, width=300),
                #Percent,
                column(p,
                width=400),    
            
        ),row(widgetbox(sdate_range_slider,width=1700),width=1500))


curdoc().add_root(layout)
curdoc().title = 'EOIs'



print(time.time()-mstart)

