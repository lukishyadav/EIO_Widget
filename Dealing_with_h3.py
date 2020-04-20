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
from bokeh.models import TextInput
from collections import Counter
from bokeh.util.hex import hexbin
from bokeh.transform import linear_cmap

from datetime import date

from bokeh.models.widgets import DateRangeSlider,DateSlider
from h3 import h3

map_repr='mercator'

infile='generated_data/rentals_wave3.csv'


import dask.dataframe as dd



max_res = 15
RESOLUTION=5

C_T=1000

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



svalue="2020-01-03 00:00:00"
evalue="2020-01-03 01:00:00"

df=df[(df['rental_started_at']>svalue) & (df['rental_started_at']<evalue)]


#df=dd.read_csv(infile)

#display_columns=df.columns







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
    
    """
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
        
    return df_aggreg


"""

def hexagons_dataframe_to_geojson(df_hex, file_output = None):
    
    '''Produce the GeoJSON for a dataframe that has a geometry column in geojson format already, along with the columns hex_id and value '''
    
    list_features = []
    
    for i,row in df_hex.iterrows():
        feature = Feature(geometry = row["geometry"] , id=row["hex_id"], properties = {"value" : row["value"]})
        list_features.append(feature)
        
    feat_collection = FeatureCollection(list_features)
    
    geojson_result = json.dumps(feat_collection)
    
    #optionally write to file
    if file_output is not None:
        with open(file_output,"w") as f:
            json.dump(feat_collection,f)
    
    return geojson_result

"""

import time

start=time.time()
df_aggreg= counts_by_hexagon(df = df, resolution = RESOLUTION)
print(time.time()-start)

"""
print(df_aggreg.shape)
df_aggreg.sort_values(by = "value", ascending = False, inplace = True)
df5=df_aggreg.head(5)
"""

df_aggreg['hexlat']=df_aggreg['center'].apply(lambda x:x['coordinates'][0][0])

df_aggreg['hexlong']=df_aggreg['center'].apply(lambda x:x['coordinates'][0][1])


cl=['hexlat', 'hexlong']

def convert_to_mercator(lngs, lats):
    projection = Proj(init='epsg:3857')
    xs = []
    ys = []
    for lng, lat in zip(lngs, lats):
        x, y = projection(lng, lat)
        xs.append(x)
        ys.append(y)
    return xs, ys


for lcol in list(range(0,len(cl),2)):    
    df_aggreg['mrc_'+cl[lcol+1]],df_aggreg['mrc_'+cl[lcol]]=convert_to_mercator(df_aggreg[cl[lcol+1]], df_aggreg[cl[lcol]])



#type(df5['geometry'].iloc[0])

#d=df5['center'].iloc[0]

purpose=df_aggreg[["mrc_hexlong", "mrc_hexlat",'value']]
maxlat=max(purpose['mrc_hexlat'])
minlat=min(purpose['mrc_hexlat'])

maxlng=max(purpose['mrc_hexlong'])
minlng=min(purpose['mrc_hexlong'])
 
purpose.columns=['q','r','counts']


purpose['C']=purpose['counts'].apply(lambda x:0 if x>C_T else 1)

#"flattop"    "pointytop"

#SIZ=1.73205080756887729352/2*(df_meta.edge_length_m.iloc[RESOLUTION])
SIZ=df_meta.edge_length_m.iloc[RESOLUTION]

from bokeh.util.hex import cartesian_to_axial
purpose['Q'],purpose['R']=cartesian_to_axial(purpose['q'], purpose['r'], SIZ, "pointytop")


#cl=['hexlat', 'hexlong']


#for lcol in list(range(0,len(cl),2)):    
#    purpose['Q'],purpose['R']=convert_to_mercator(purpose['Q'], purpose['R'])


purpose_source = ColumnDataSource()

dictionary = dict(
    Q=purpose['Q'],
    R=purpose['R'],
    C=purpose['C'],
    counts=purpose['counts']    
    #label=datapoints_df['label'],
    #time=datapoints_df['start_datetime']
    )

purpose_source.data = dictionary 

from bokeh.tile_providers import CARTODBPOSITRON 




map_repr='mercator'
# set up/draw the map
p = figure(
    x_range=(minlng,maxlng),
    y_range=(minlat, maxlat),
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




display_columns2=purpose.columns
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

p.hex_tile(q="Q", r="R", size=df_meta.edge_length_m.iloc[RESOLUTION], line_color=None, source=purpose_source,
           
           fill_color=linear_cmap('C', RdYlGn[5], 0, max(purpose.C)),line_alpha=0,fill_alpha=0.4)

from bokeh.io import output_file, show 

output_file("hex_tile.html")


#VERIFICATION STUFF
datapoints_source = ColumnDataSource()

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

p.circle(x='x', y='y', 
                  #size=cluster_point_size,
                  fill_alpha=1,
                  source=datapoints_source,color="royalblue",line_alpha=0
                  #line_color='black'
                  )


show(p)  






"""


import numpy as np

from bokeh.io import output_file, show
from bokeh.models import HoverTool
from bokeh.plotting import figure

n = 500
x = 2 + 2*np.random.standard_normal(n)
y = 2 + 2*np.random.standard_normal(n)

p = figure(title="Hexbin for 500 points", match_aspect=True,
           tools="wheel_zoom,reset", background_fill_color='#440154')
p.grid.visible = False

r, bins = p.hexbin(x, y, size=0.5, hover_color="pink", hover_alpha=0.8)

p.circle(x, y, color="white", size=1)

p.add_tools(HoverTool(
    tooltips=[("count", "@c"), ("(q,r)", "(@q, @r)")],
    mode="mouse", point_policy="follow_mouse", renderers=[r]
))

output_file("hexbin.html")

show(p)







import numpy as np

from bokeh.io import output_file, show
from bokeh.plotting import figure
from bokeh.transform import linear_cmap
from bokeh.util.hex import hexbin

n = 50000
x = np.random.standard_normal(n)
y = np.random.standard_normal(n)

bins = hexbin(x, y, 0.1)

p = figure(title="Manual hex bin for 50000 points", tools="wheel_zoom,pan,reset",
           match_aspect=True, background_fill_color='#440154')
p.grid.visible = False

p.hex_tile(q="q", r="r", size=0.1, line_color=None, source=bins,
           fill_color=linear_cmap('counts', 'Viridis256', 0, max(bins.counts)),orientation='pointy_top')

output_file("hex_tile.html")

show(p)

"""

print(time.time()-mstart)

