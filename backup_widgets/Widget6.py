#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 10:02:40 2020

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
from math import radians, cos, sin, asin, sqrt

map_repr='mercator'

#infile='generated_data/rentals_wave3.csv'
#infile='daytona_rental_data.csv'
infile='journey_data/eiffel_rental_data.csv'

#PRESETS

max_res = 15

RESOLUTION=7

C_T=1


svalue=str(datetime.datetime(2019,8,15,0))
evalue=str(datetime.datetime(2019,8,15,6))



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
#df2=pd.read_csv(infile)





#df=df[(df['rental_started_at']>svalue) & (df['rental_started_at']<evalue)]



#df=df[(df['rental_started_at']>svalue) & (df['rental_started_at']<evalue)]


df= df[(df['rental_started_at']>svalue) & (df['rental_started_at']<evalue)]




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


cl=['start_lat', 'start_long', 'end_lat','end_long']


for lcol in list(range(0,len(cl),2)):    
    df['mrc_'+cl[lcol+1]],df['mrc_'+cl[lcol]]=convert_to_mercator(df[cl[lcol+1]], df[cl[lcol]])
    
    
latlong=['mrc_start_lat','mrc_start_long']

latlong=['start_lat','start_long']

def counts_by_hexagon(df, resolution,latlong):    
    '''Use h3.geo_to_h3 to index each data point into the spatial index of the specified resolution.
      Use h3.h3_to_geo_boundary to obtain the geometries of these hexagons'''

    #df = df[["latitude","longitude"]]
    df=df[latlong]
    print('1st')
    #df["hex_id"] = df.apply(lambda row: h3.geo_to_h3(row["latitude"], row["longitude"], resolution), axis = 1)
    df["hex_id"] = df.apply(lambda row: h3.geo_to_h3(row[latlong[0]], row[latlong[1]], resolution), axis = 1)
    
    df_aggreg = df.groupby(by = "hex_id").size().reset_index()
    print(len(df_aggreg))
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
df_aggreg= counts_by_hexagon(df,RESOLUTION,latlong)
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
    #carsharing_text.text='start'
    import time
    s=time.time()
    pre.text='<h4 style="border-top: 2px solid #778899;width: 1600px"><br><b style="color:slategray">Update in Progress....</b><br></h4>'
    print(time.time()-s)
    #print(toggle_small.active)
    range_slider1=sdate_range_slider
    RESOLUTION=resolution_slider.value
    C_T=int(count_threshold.value)



    df=pd.read_csv(infile)
    
    
    for lcol in list(range(0,len(cl),2)):    
        df['mrc_'+cl[lcol+1]],df['mrc_'+cl[lcol]]=convert_to_mercator(df[cl[lcol+1]], df[cl[lcol]])
    
    if se_select.value=='Start':
       latlong=['start_lat','start_long']
       df['x']=df['mrc_start_long']
       df['y']=df['mrc_start_lat']
    else:    
       latlong=['end_lat','end_long']
       df['x']=df['mrc_end_long']
       df['y']=df['mrc_end_lat']
    
    
    
    
    
        
    if isinstance(range_slider1.value[0], (int, float)):
    # pandas expects nanoseconds since epoch
        start_date = pd.Timestamp(float(range_slider1.value[0])*1e6)
        end_date = pd.Timestamp(float(range_slider1.value[1])*1e6)
    else:
        start_date = pd.Timestamp(range_slider1.value[0])
        end_date = pd.Timestamp(range_slider1.value[1])
    

    start_date=Mindate+timedelta(hours=date_widget.value[0])
    end_date=Mindate+timedelta(hours=date_widget.value[1])
    
    
    
    print(start_date,end_date)
    
    ms=max(df['rental_started_at'])
    
    df=df[(df['rental_started_at']>str(start_date)) & (df['rental_started_at']<str(end_date))]
    
    start=time.time()
    df_aggreg= counts_by_hexagon(df,RESOLUTION,latlong)
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
        
    
    #datapoints_source = ColumnDataSource(df)
    if toggle_path.active==True:
        source.data=dict(
        x=[],
        y=[],
        x1=[],
        y1=[],
        cx=[],
        cy=[],
    )

    else:    
        df['haversine_distance']=df.apply(haversine,axis=1)
        source.data=dict(
        x=df['mrc_start_long'],
        y=df['mrc_start_lat'],
        x1=df['mrc_end_long'],
        y1=df['mrc_end_lat'],
        cx=(df['mrc_start_long']+df['mrc_end_long'])/2,
        cy=df['mrc_start_lat']+df['haversine_distance']/8,
    )


    #th=list(set(purpose['C']))
    #th=[str(i) for i in th]
    #mapper['transform'].factors=th
    
    #AttributeError: unexpected attribute 'factors' to LinearColorMapper, possible attributes are high, high_color, js_event_callbacks, js_property_callbacks, low, low_color, name, nan_color, palette, subscribed_events or tags
    
    maxlat=max(df['mrc_start_lat'])
    minlat=min(df['mrc_start_lat'])
    
    maxlng=max(df['mrc_end_long'])
    minlng=min(df['mrc_end_long'])
    
    if span_radio.active==1:  
        p.toolbar.active_tap=None
        p.x_range.end=maxlng
        p.x_range.start=minlng
        p.y_range.end=maxlat
        p.y_range.start=minlat

    pre.text='<h4 style="border-top: 2px solid #778899;width: 1600px"><br><b style="color:slategray">High Utilization Areas</b><br>'+'<b style="color:slategray">Filtered Idle spots: </b>'+str(len(df))+'</h4>'
            



def update_click():
    pre.text='<h4 style="border-top: 2px solid #778899;width: 1600px"><br><b style="color:slategray">Update in Progress....</b><br></h4>'
    curdoc().add_next_tick_callback(my_slider_handler)
    
"""
WIDGETS

"""

sdate_input = TextInput(value="2020-01-03 00:00:00", title="Start Date: (YYYY-MM-DD HH:MM:SS)")
#sdate_input.on_change("value", my_slider_handler)

edate_input = TextInput(value="2020-01-03 01:00:00", title="End Date: (YYYY-MM-DD HH:MM:SS)")
#edate_input.on_change("value", my_slider_handler)
count_threshold = TextInput(value="1", title="Count Threshold Value")

resolution_slider=Slider(start=1, end=15, value=7, step=1, title="H3 Resolutions")

sdate_range_slider = DateRangeSlider(title="Date Range: ", start=datetime.datetime(2019, 8, 15,0), end=datetime.datetime(2020, 4, 28,1), value=(datetime.datetime(2020, 1, 3,1), datetime.datetime(2020, 1, 3,2)),format="%x,%X")

#sdate_range_slider = DateRangeSlider(title="Date Range: ", start=datetime.datetime(2017, 1, 1,1), end=datetime.datetime(2017, 2, 7,2), value=(datetime.datetime(2017, 9, 7,1), datetime.datetime(2017, 9, 7,2)),format="%Y-%m-%d %H")
#sdate_range_slider = DateRangeSlider(title="Date Range: ", start=datetime.datetime(2017, 1, 1,1), end=datetime.datetime(2017, 2, 7,2), value=(datetime.datetime(2017, 9, 7,1), datetime.datetime(2017, 9, 7,2)),step=1)
#sdate_range_slider.on_change("value", my_slider_handler)
#sdate_range_slider = DateSlider(title="Date Range: ", start=datetime.datetime(2017, 1, 1,1), end=datetime.datetime(2019, 9, 7,2), value=(datetime.datetime(2017, 9, 7,1), datetime.datetime(2017, 9, 7,2)), step=1)

sMindate=datetime.datetime(2020, 1,3,0)

Mindate=datetime.datetime(2019,8,15,0)

date_text = Div(text='<b style="color:black">'+str(Mindate)+'&nbsp;&nbsp;&nbsp;to&nbsp;&nbsp;&nbsp;'+str(Mindate+timedelta(hours=6))+'<br></b>',width=500, height=40)

def date_function(attr, old, new):
    NMindate=Mindate+timedelta(hours=date_widget.value[0])
    NMaxdate=Mindate+timedelta(hours=date_widget.value[1])
    
    #fNMindate=Mindate+timedelta(hours=fine_date_widget.value[0])
    #fNMaxdate=Mindate+timedelta(hours=fine_date_widget.value[1])
    
    date_text.text='<b style="color:black">'+str(NMindate)+'&nbsp;&nbsp;&nbsp;to&nbsp;&nbsp;&nbsp;'+str(NMaxdate)+'<br></b>'
    
    #fine_date_text.text='<b style="color:black">'+str(fNMindate)+'&nbsp;&nbsp;&nbsp;to&nbsp;&nbsp;&nbsp;'+str(fNMaxdate)+'<br></b>'

date_widget = RangeSlider(start=0, end=5760, value=(0,6), step=1,show_value=False,tooltips=False)

date_widget.on_change('value', date_function)


se_select = Select(title="Start/End Locations", value="Start", options=['Start','End'])

span_radio=RadioButtonGroup(
        labels=["Lock Map Area","Auto Adjust Map Area"], active=0)

bt = Button(label='Update Plot')
bt.on_click(update_click)



alpha_range_slider = Slider(start=0, end=1, value=0.4, step=.1, title="Spot Transparency")

size_range_slider = Slider(start=4, end=50, value=4, step=1, title="Spot Size")

def alpha_size(attr, old, new):
    circle_plot.glyph.size=size_range_slider.value
    
    circle_plot.glyph.fill_alpha=alpha_range_slider.value
    

alpha_range_slider.on_change('value', alpha_size)

size_range_slider.on_change('value', alpha_size)


def toggle_small_handler(attr, old, new):
    print(toggle_small.active)
    my_slider_handler()
    
def toggle_hexes_handler(attr, old, new):
    print(toggle_hexes.active)
    my_slider_handler()


def toggle_path_handler(attr, old, new):
    print(toggle_path.active)
    my_slider_handler()


toggle_small = Toggle(label="Toggle Points", button_type="success")
toggle_small.on_change("active",  toggle_small_handler)


toggle_hexes = Toggle(label="Toggle Hexes", button_type="success")
toggle_hexes.on_change("active",  toggle_hexes_handler)


toggle_path = Toggle(label="Toggle Path", button_type="success")
toggle_path.on_change("active",  toggle_path_handler)


CID=list(set(df['customer_id'].astype('str')))
print('CID',len(CID))
select_cid = Select(title="Select Customer IDs:", value="foo", options=CID)


global pre 

pre = Div(text='<h4 style="border-top: 2px solid #778899;width: 1600px"><br><b style="color:slategray">High Utilization Areas</b><br>'+'<b style="color:slategray">Filtered Idle spots: </b>'+str(len(df))+'</h4>',
width=500, height=50)



#sfrom bokeh.tile_providers import CARTODBPOSITRON 

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


tiles = {'Light':get_provider(CARTODBPOSITRON),'Open Street':get_provider(OSM),'Satellite':get_provider(ESRI_IMAGERY)}
#select menu

#callback
def change_tiles_callback(attr, old, new):
    #removing the renderer corresponding to the tile layer
    p.renderers = [x for x in p.renderers if not str(x).startswith('TileRenderer')]
    #inserting the new tile renderer
    tile_renderer = TileRenderer(tile_source=tiles[new])
    p.renderers.insert(0, tile_renderer)
 

tile_prov_select = Select(title="Tile Provider", value="Light", options=['Light','Open Street','Satellite'])

    
tile_prov_select.on_change('value',change_tiles_callback) 



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
df['x']=df['mrc_start_long']
df['y']=df['mrc_start_lat']
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
circle_plot=p.circle(x='x', y='y', 
                  #size=cluster_point_size,
                  fill_alpha=0.2,
                  source=datapoints_source,color="royalblue",line_alpha=0.4
                  #line_color='black'
                  )

#show(p)
df['haversine_distance']=df.apply(haversine,axis=1)

source = ColumnDataSource(dict(
        x=df['mrc_start_long'],
        y=df['mrc_start_lat'],
        x1=df['mrc_end_long'],
        y1=df['mrc_end_lat'],
        cx=(df['mrc_start_long']+df['mrc_end_long'])/2,
        cy=df['mrc_start_lat']+df['haversine_distance']/8,
    )
)



glyph = p.quadratic(x0="x", y0="y", x1="x1", y1="y1",cx='cx',cy='cy', line_color="darkslategrey", line_width=2,source=source)


carsharing_text = Div(text='<h2 style="color:darkslategray;font-family: "Lucida Console", Courier, monospace;">Carsharing Journey Tracking Tool</h2>',
width=500, height=40)

     
layout = column(row(carsharing_text,height=70),row(
            column(
                    row(widgetbox(pre),height=100),
                widgetbox(select_cid),
                widgetbox(count_threshold),   
                widgetbox(resolution_slider),
                widgetbox(toggle_small),
                widgetbox(toggle_hexes),
                widgetbox(toggle_path),
                widgetbox(tile_prov_select),
                widgetbox(se_select),
                widgetbox(span_radio),
                widgetbox(alpha_range_slider),
                widgetbox(size_range_slider),
                widgetbox(bt)
                #widgetbox(hovertool_widget)
                ,width=400),   
                #widgetbox(slider,width=350),
                #widgetbox(Min_n, width=300),
                #Percent,
                column(row(height=100),p,
                width=400),    
            
        ),row(widgetbox(date_text)),
                row(widgetbox(date_widget,width=1400),width=1500))


curdoc().add_root(layout)
curdoc().title = 'EOIs'



print(time.time()-mstart)

