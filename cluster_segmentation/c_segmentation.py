#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 16:38:21 2020

@author: lukishyadav
"""

from bokeh.io import curdoc
import inspect
import json
#import logging
from bokeh.layouts import column,layout,row,widgetbox
import pandas as pd
import datetime
from bokeh.plotting import figure, show, output_file, ColumnDataSource
from bokeh.transform import factor_cmap
from bokeh.tile_providers import CARTODBPOSITRON
import numpy as np
from bokeh.models.widgets import Button, RadioButtonGroup, Select, Slider,TextInput
from bokeh.models import MultiSelect
from sklearn.cluster import KMeans,DBSCAN, MeanShift,AgglomerativeClustering
from bokeh.models import TextAreaInput

from bokeh.models.annotations import Title
import pandas as pd
import datetime
from numpy import mean
import collections
from collections import Counter as c
from bokeh.models import CheckboxGroup



from bokeh.server.server import Server
from bokeh.application import Application
from bokeh.application.handlers.function import FunctionHandler
from bokeh.plotting import figure, ColumnDataSource
from bokeh.models import Button,Tabs,Panel

from bokeh.models.widgets import PreText
from bokeh.models import Div

import os
from bokeh.models import ColumnDataSource, HoverTool, TapTool, PolySelectTool,LassoSelectTool,BoxSelectTool
from bokeh.events import Tap,SelectionGeometry,Pan,PanStart,PanEnd
#sub='segmentation/'


from bokeh.models import (Button, ColumnDataSource, CustomJS, DataTable,
                          NumberFormatter, RangeSlider, TableColumn,)

from os.path import dirname, join

from sklearn.preprocessing import MinMaxScaler,StandardScaler,Normalizer
from bokeh.models import ColumnDataSource, HoverTool, TapTool, PolySelectTool,LassoSelectTool,BoxSelectTool
map_repr='mercator'
from bokeh.events import Tap,SelectionGeometry,Pan,PanStart,PanEnd
sub=''
global df,report_data
from bokeh.layouts import gridplot
id_column='customer_id'
df = pd.read_csv(sub+'Darwin_the_data.csv')

selected_features=[]


table_height=300
table_width=1200

#Description dataframe
fd=df.describe().T.reset_index()
fd=fd.round(decimals=2)
dsc_table_Columns = [TableColumn(field=Ci, title=Ci) for Ci in fd.columns] # bokeh columns
dsc_table_source=ColumnDataSource(fd.round(decimals=2))
dsc_table = DataTable(columns=dsc_table_Columns, source=dsc_table_source,width=table_width,height=table_height)
dsc_panel=Panel(child=column(row(dsc_table,width=table_width,height=table_height)),title='Input Data Description')


input_table_Columns = [TableColumn(field=Ci, title=Ci) for Ci in df.columns] 
input_table_source=ColumnDataSource(df.round(decimals=2))
input_table = DataTable(columns=input_table_Columns, source=input_table_source,width=table_width,height=table_height)
input_panel=Panel(child=column(row(input_table,width=table_width,height=table_height)),title='Input Data Table')

def Update_Click():
    print(multi_select.value)
    print(list(fd.iloc[dsc_table_source.selected.indices,0].values))
    print(type(json.loads(input_arguments.value)))
    param_dict=json.loads(input_arguments.value)
    if select_scaler.value!='None':
        scaler_param_dict=json.loads(scaler_input_arguments.value)
    #Option to select columns(data) from data table source of bokeh.
    selected_columns=list(fd.iloc[dsc_table_source.selected.indices,0].values)
    selected_columns=multi_select.value
    
    if selected_columns !=[]:
        selected_data=df[selected_columns].copy()
        print(select_algorithm.value)
        if select_scaler.value!='None':
            selected_scaler=scaler_dict[select_scaler.value]
        selected_algorithm=algorithm_dict[select_algorithm.value]
        
        if cluster_preset_checkbox.active!=[]:
            
            from sklearn.metrics import silhouette_score
        
            sil = []
            kmax = 10
            
            c_range=list(range(2, kmax+1))
            # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
            for k in range(2, kmax+1):
              sa = selected_algorithm(n_clusters = k).fit(selected_data.values)
              labels = sa.labels_
              sil.append(silhouette_score(selected_data.values, labels, metric = 'euclidean'))
        
            param_dict['n_clusters']=c_range[sil.index(max(sil))]



        if select_scaler.value=='None':
            clus = selected_algorithm(**param_dict).fit(selected_data.values)
        else:
            clus= selected_algorithm(**param_dict).fit(selected_scaler(**scaler_param_dict).fit_transform(selected_data.values))    

        print(clus.labels_)
        output_data=selected_data.copy()
        output_data['Cluster']=clus.labels_
        #output_data[id_column]=df[id_column]
        

        sd_table_Columns = [TableColumn(field=Ci, title=Ci) for Ci in selected_data.columns]
        sd_table_source.data=selected_data.round(decimals=2)
        sd_table.columns=sd_table_Columns
        sd_table.source=sd_table_source
        
        output_table_Columns = [TableColumn(field=Ci, title=Ci) for Ci in output_data.columns] 
        output_table_source.data=output_data.round(decimals=2)
        output_table.columns=output_table_Columns
        output_table.source=output_table_source
        
        output_data['for_count']=[1 for i in range(len(output_data))]
        print('output data len',len(output_data))
        
        
        groupby_dict={}
        for v in selected_columns:
                 groupby_dict[v]='mean'
        groupby_dict['for_count']='count' 
                 
        current_values_output=['AVERAGE_OF_'+cv for cv in selected_columns]   
        display_columns=['Cluster','Count']
        display_columns.extend(current_values_output)
        cvo=['Cluster']
        cvo.extend(current_values_output)
        cvo.append('Count')  
        report_data=output_data.groupby('Cluster').agg(groupby_dict).reset_index()
        report_data.columns=cvo
        report_data=report_data[display_columns]
        report_table_Columns = [TableColumn(field=Ci, title=Ci) for Ci in report_data.columns] 
        report_table_source.data=report_data.round(decimals=2)
        report_table.columns=report_table_Columns
        report_table.source=report_table_source
        
        
        

        
    





#multiselect

raw_columns=list(df.columns)
option_list=[]
for col in raw_columns:
    option_list.append((col,col))
    
current_values=['Total_fare(Total of all bills)',
       'Total_revenue(Total of actual fees paid)', 'Days_from_last_rental',
       'Avg_rental_duration (excluding park time)',
       'Average_time_between_rental(including park time)',
       'Average_bill_per_trip']
multi_select = MultiSelect(title="Select Features", value= current_values,options= option_list,width=400)



selected_data=df[current_values].copy()
selected_data=selected_data.round(decimals=2)
sd_table_Columns = [TableColumn(field=Ci, title=Ci) for Ci in selected_data.columns] # bokeh columns
sd_table_source=ColumnDataSource(selected_data.round(decimals=2))
sd_table = DataTable(columns=sd_table_Columns, source=sd_table_source,width=table_width,height=table_height)


sd_panel=Panel(child=column(row(sd_table,width=table_width,height=table_height)),title='Selected Data Table')



#Clustering_Algorithm
cluster_functions=[KMeans,DBSCAN, MeanShift,AgglomerativeClustering]
cluster_options=['KMeans','DBSCAN','MeanShift','AgglomerativeClustering']
algorithm_dict={}
for k in range(len(cluster_options)):
    algorithm_dict[cluster_options[k]]=cluster_functions[k]
select_algorithm = Select(title="Select Clustering Algorithm", value=cluster_options[0], options=cluster_options,width=200)

def algo_change_function(attr,old,new):
    #arg_text.text='<b style="border:2px LightGray;border-radius:25px;border-style:groove;padding:5px">'+str(inspect.signature(algorithm_dict[select_algorithm.value]))+'</b>'
    model_inspect=inspect.getargspec(algorithm_dict[select_algorithm.value])
    inspect_list=model_inspect[0][1:]
    inspect_values=model_inspect[3]
    inspect_dict={}
    for k in range(len(inspect_values)):
        inspect_dict[inspect_list[k]]=inspect_values[k]

    param_dict=json.dumps(inspect_dict)

    input_arguments.value=param_dict
    print(input_arguments.value)

select_algorithm.on_change('value',algo_change_function)
    



#scalers_selection
scaler_functions=[MinMaxScaler,StandardScaler,Normalizer]
scaler_options=['MinMaxScaler','StandardScaler','Normalizer']
scaler_dict={}
for k in range(len(scaler_options)):
    scaler_dict[scaler_options[k]]=scaler_functions[k]
  
scaler_options.append('None')
select_scaler = Select(title="Select Scaling", value='None', options=scaler_options,width=200)

def scaler_change_function(attr,old,new):
    #arg_text.text='<b style="border:2px LightGray;border-radius:25px;border-style:groove;padding:5px">'+str(inspect.signature(scaler_dict[select_scaler.value]))+'</b>'
    if select_scaler.value!='None':
        scaler_inspect=inspect.getargspec(scaler_dict[select_scaler.value])
        scaler_inspect_list=scaler_inspect[0][1:]
        scaler_inspect_values=scaler_inspect[3]
        scaler_inspect_dict={}
        for k in range(len(scaler_inspect_values)):
            scaler_inspect_dict[scaler_inspect_list[k]]=scaler_inspect_values[k]
    
        scaler_param_dict=json.dumps(scaler_inspect_dict)
    
        scaler_input_arguments.value=scaler_param_dict
        print(scaler_input_arguments.value)
    else:
        scaler_input_arguments.value=''

select_scaler.on_change('value',scaler_change_function)

    

#defauly arguments/parameters text
#arg_text = Div(text='<b style="border:2px LightGray;border-radius:25px;border-style:groove;padding:5px">'+str(inspect.signature(algorithm_dict[cluster_options[0]]))+'</b>',width=2000, height=100)

model_inspect=inspect.getargspec(algorithm_dict[cluster_options[0]])
inspect_list=model_inspect[0][1:]
inspect_values=model_inspect[3]
inspect_dict={}
for k in range(len(inspect_values)):
    inspect_dict[inspect_list[k]]=inspect_values[k]

param_dict=json.dumps(inspect_dict)

input_arguments=TextAreaInput(value=param_dict, rows=10, title="Algorithm Arguments dictionary",width=200)



#scaler_arg_text = Div(text='<b style="border:2px LightGray;border-radius:25px;border-style:groove;padding:5px">'+'</b>',width=2000, height=100)


scaler_input_arguments=TextAreaInput(value='', rows=10, title="Scaler Arguments dictionary",width=200)

if select_scaler.value!=None:
    scaler_inspect=inspect.getargspec(scaler_dict[scaler_options[0]])
    scaler_list=scaler_inspect[0][1:]
    scaler_values=scaler_inspect[3]
    scaler_inspect_dict={}
    for k in range(len(scaler_values)):
        scaler_inspect_dict[scaler_list[k]]=scaler_values[k]
    
    scaler_param_dict=json.dumps(scaler_inspect_dict)
else:    
    scaler_input_arguments.value=''



param_dict=json.loads(param_dict)
scaler_param_dict=json.loads(scaler_param_dict)

"""







CLUSTER PRESET CHANGE APPLIED





"""


cluster_preset_checkbox = CheckboxGroup(
        labels=["Cluster Preset"], active=[0])


if select_scaler.value!='None':
    selected_scaler=scaler_dict[select_scaler.value]    
selected_algorithm=algorithm_dict[select_algorithm.value]

if cluster_preset_checkbox.active!=[]:
    
    from sklearn.metrics import silhouette_score

    sil = []
    kmax = 10
    
    c_range=list(range(2, kmax+1))
    # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
    for k in range(2, kmax+1):
      sa = selected_algorithm(n_clusters = k).fit(selected_data.values)
      labels = sa.labels_
      sil.append(silhouette_score(selected_data.values, labels, metric = 'euclidean'))

    param_dict['n_clusters']=c_range[sil.index(max(sil))]



if select_scaler.value=='None':
    clus = selected_algorithm(**param_dict).fit(selected_data.values)
else:
    clus= selected_algorithm(**param_dict).fit(selected_scaler(**scaler_param_dict).fit_transform(selected_data.values))    
output_data=selected_data.copy()
output_data['Cluster']=clus.labels_



#output_data=selected_data.copy()
output_data=output_data.round(decimals=2)
output_table_Columns = [TableColumn(field=Ci, title=Ci) for Ci in output_data.columns] 
output_table_source=ColumnDataSource(output_data.round(decimals=2))
output_table = DataTable(columns=output_table_Columns, source=output_table_source,width=table_width,height=table_height)
output_panel=Panel(child=column(row(output_table,width=table_width,height=table_height)),title='Output Data Table')



output_data['for_count']=[1 for i in range(len(output_data))]

groupby_dict={}
for v in current_values:
         groupby_dict[v]='mean'
groupby_dict['for_count']='count' 

current_values_output=['AVERAGE_OF_'+cv for cv in current_values]  
cvo=['Cluster']
cvo.extend(current_values_output)
cvo.append('Count')  

display_columns=['Cluster','Count']
display_columns.extend(current_values_output)
report_data=output_data.groupby('Cluster').agg(groupby_dict).reset_index()
report_data.columns=cvo
report_data=report_data[display_columns]
report_table_Columns = [TableColumn(field=Ci, title=Ci) for Ci in report_data.columns] 
report_table_source=ColumnDataSource(report_data.round(decimals=2))
report_table = DataTable(columns=report_table_Columns, source=report_table_source,width=table_width,height=table_height)
report_panel=Panel(child=column(row(report_table,width=1000)),title='Report Data Table')



first_hist_selected=df._get_numeric_data().columns[5]

hist, edges = np.histogram(df[first_hist_selected], density=True, bins=50)
                   
hist_df = pd.DataFrame({"column": hist,
                        "left": edges[:-1],
                        "right": edges[1:]})
hist_df["interval"] = ["%d to %d" % (left, right) for left, 
                        right in zip(hist_df["left"], hist_df["right"])]


src = ColumnDataSource(hist_df)


hist_plot = figure(tools="pan,wheel_zoom,box_zoom,tap,box_select,reset,save", background_fill_color="#fafafa",height=400,width=400)
t = Title()
t.text =first_hist_selected+' Histogram'
hist_plot.title=t               
hist_quad=hist_plot.quad(top="column", bottom=0, left='left', right='right',source = src,
   fill_color="navy", line_color="white", alpha=0.5)
hist_plot.y_range.start = 0
hist_plot.legend.location = "center_right"
hist_plot.legend.background_fill_color = "#fefefe"
hist_plot.xaxis.axis_label = 'x'
hist_plot.yaxis.axis_label = 'Pr(x)'
hist_plot.grid.grid_line_color="white"


def hist_func(attr,old,new):
    global hist_df
    print(list(fd.iloc[dsc_table_source.selected.indices,0].values))
    first_hist_selected=list(fd.iloc[dsc_table_source.selected.indices,0].values)[0]
    hist, edges = np.histogram(df[first_hist_selected], density=True, bins=50)
                   
    hist_df = pd.DataFrame({"column": hist,
                            "left": edges[:-1],
                            "right": edges[1:]})
    hist_df["interval"] = ["%d to %d" % (left, right) for left, 
                            right in zip(hist_df["left"], hist_df["right"])]
    
    
    src.data=hist_df
    
    t.text =first_hist_selected+' Histogram'
    hist_plot.title=t               

    
taptool = hist_plot.select(type=TapTool)

def hist_bt_func():
    global df
    print(list(hist_df.iloc[src.selected.indices,:].values))
    f_hist_df=hist_df.iloc[src.selected.indices,:]
    left_min=min(f_hist_df['left'])
    right_max=max(f_hist_df['right'])
    first_hist_selected=list(fd.iloc[dsc_table_source.selected.indices,0].values)[0]
    print(left_min,right_max)
    df=df[(df[first_hist_selected]>=left_min) & (df[first_hist_selected]<=right_max)]
    input_table_source.data=df
    FD=df.describe().T.reset_index()    
    dsc_table_source.data=FD
    #input_table_source=ColumnDataSource(df)   
  
hist_plot.on_event(Tap,hist_bt_func)

    
hist_bt = Button(label='Apply Changes',button_type='primary')
hist_bt.on_click(hist_bt_func)


dsc_table.source.selected.on_change('indices',hist_func)

#arg_text = Div(text='<b style="border:2px LightGray,border-radius:25px,border-style:ridge,padding:5px">'+'1'+'</b>',width=200, height=300)

#arg_text = Div(text='<b style="border-width:5px;border-style:groove;">'+str(inspect.signature(algorithm_dict[cluster_options[0]]))+'</b>',width=200, height=300)

#Update Button
#default, primary, success, warning, danger
update_bt = Button(label='Update Plot',button_type='primary')
update_bt.on_click(Update_Click)


def Reset_Click():
    df=pd.read_csv(sub+'Darwin_the_data.csv')  
    fd=df.describe().T.reset_index()
    dsc_table_source.data=fd 
    input_table_source.data=df

reset_bt =Button(label='Reset Input Data',button_type='warning')
reset_bt.on_click(Reset_Click)



"""


SEGMENTATION SECTION



"""


OD=output_data[output_data['Cluster']==1]

global slider

CL=list(OD._get_numeric_data().columns)

X=OD._get_numeric_data().columns[0]

Y=OD._get_numeric_data().columns[1]

Z=OD._get_numeric_data().columns[2]


threhsold_slider = Slider(start=min(OD[Z]), end=max(OD[Z]), value=1, step=1, title="Color Threshold (For determining color of points)")

Xselect = Select(title="X-axis", value=X, options=CL)
#Xselect.on_change('value', my_slider_handler)

Yselect = Select(title="Y-axis", value=Y, options=CL)
#Yselect.on_change('value', my_slider_handler)

Zselect = Select(title="Color Threshold Column", value=Z, options=CL)
#Zselect.on_change('value', my_slider_handler)

seg_plot = figure(tools="pan,wheel_zoom,box_zoom,box_select,reset,save",height=400,width=400)


OD['thr']=OD[Z].apply(lambda x:1 if x>threhsold_slider.value else 0)
OD['thr']=OD['thr'].astype('str')
OD['x']=OD[X]
OD['y']=OD[Y]

seg_dictionary={}

for col_name in OD.columns:
# if col_name not in [X,Y]:
  seg_dictionary[col_name]=OD[col_name]

datapoints_source = ColumnDataSource()
datapoints_source.data = seg_dictionary


th=list(set(OD['thr']))
mapper=factor_cmap('thr', 'Category10_3', th)

st = Title()
st.text = "Segmentation Plot"
seg_plot.title=st

seg_plot.scatter("x", "y", source=datapoints_source, legend="thr", alpha=0.5,size=12, color=mapper)
seg_plot.xaxis.axis_label = X
seg_plot.yaxis.axis_label = Y
legend_title = Z+"_Threshold" #Legend Title
seg_plot.legend.title = '> Threshold is 1 else 0 (For Numeric Columns)'


m_factors=mapper['transform'].factors
one_index=m_factors.index('1')

m_palette=mapper['transform'].palette

LINE_ARGS = dict(color="#3A5785", line_color=None)
LINE_ARGS = dict(color=m_palette[one_index],line_color=None)                 
                 

def hist_variables_create(data,edge_override=None):
    # create the horizontal histogram
    
    if edge_override is not None:
        hhist, hedges = np.histogram(data, bins=edge_override)
    else:
        hhist, hedges = np.histogram(data, bins=20)
    hzeros = np.zeros(len(hedges)-1)
    hmax = max(hhist)*1.1
    
    h_hist_df = pd.DataFrame({"column": hhist,
                            "left": hedges[:-1],
                            "right": hedges[1:],
                            "zeros":hzeros})
    
        
    h_hist_df["interval"] = ["%d to %d" % (left, right) for left, 
                            right in zip(h_hist_df["left"], h_hist_df["right"])]
    
    return hmax,hedges,h_hist_df


hmax,hedges,h_hist_df=hist_variables_create(OD[X])      

hmax1,hedges1,h_hist_df1=hist_variables_create(OD[X][OD['thr']=='1'],hedges)            

ph = figure(toolbar_location=None, plot_width=seg_plot.plot_width, plot_height=200, x_range=seg_plot.x_range,
            y_range=(-hmax, hmax), min_border=10, min_border_left=50, y_axis_location="right")
ph.xgrid.grid_line_color = None
ph.yaxis.major_label_orientation = np.pi/4
ph.background_fill_color = "#fafafa"



h_hist = ColumnDataSource(h_hist_df)
h_hist1 = ColumnDataSource(h_hist_df1)

ph.quad(bottom=0, left='left', right='right', top='column', color="white", line_color="#3A5785",source=h_hist)
hh1 = ph.quad(bottom=0, left='left', right='right', top='column', alpha=0.5, **LINE_ARGS,source=h_hist1)
#hh2 = ph.quad(bottom=0, left=hedges[:-1], right=hedges[1:], top='column', alpha=0.1, **LINE_ARGS)

vmax,vedges,v_hist_df=hist_variables_create(OD[Y])

vmax1,vedges1,v_hist_df1=hist_variables_create(OD[Y][OD['thr']=='1'],vedges)

pv = figure(toolbar_location=None, plot_width=200, plot_height=seg_plot.plot_height, x_range=(-vmax, vmax),
            y_range=seg_plot.y_range, min_border=10, y_axis_location="right")
pv.ygrid.grid_line_color = None
pv.xaxis.major_label_orientation = np.pi/4
pv.background_fill_color = "#fafafa"



"""
# create the vertical histogram
vhist, vedges = np.histogram(OD[Y], bins=20)
vzeros = np.zeros(len(vedges)-1)
vmax = max(vhist)*1.1

v_hist_df = pd.DataFrame({"column": vhist,
                        "left": vedges[:-1],
                        "right": vedges[1:],
                        "zeros":vzeros})

    
v_hist_df["interval"] = ["%d to %d" % (left, right) for left, 
                        right in zip(v_hist_df["left"], v_hist_df["right"])]

"""



v_hist = ColumnDataSource(v_hist_df)
v_hist1 = ColumnDataSource(v_hist_df1)



pv.quad(left=0, bottom='left', top='right', right='column', color="white", line_color="#3A5785",source=v_hist)
vh1 = pv.quad(left=0, bottom='left', top='right', right='column', alpha=0.5, **LINE_ARGS,source=v_hist1)
#vh2 = pv.quad(left=0, bottom=vedges[:-1], top=vedges[1:], right=vzeros, alpha=0.1, **LINE_ARGS)

joint_plot=layout = gridplot([[seg_plot, pv], [ph, None]], merge_tools=False)







def segment_func():
    global report_data
    print(report_table_source.selected.indices)
    print(list(set(report_data.iloc[report_table_source.selected.indices,0])))
    global output_data
    OD=output_data.copy()
    OD=OD[OD['Cluster'].isin(list(set(report_data.iloc[report_table_source.selected.indices,0])))]
    print(set(OD['Cluster']))
    X=Xselect.value
    Y=Yselect.value
    Z=Zselect.value
    OD['thr']=OD[Z].apply(lambda x:1 if x>threhsold_slider.value else 0)
    OD['thr']=OD['thr'].astype('str')
    OD['x']=OD[X]
    OD['y']=OD[Y]
    seg_dictionary={}

    for col_name in OD.columns:
    # if col_name not in [X,Y]:
      seg_dictionary[col_name]=OD[col_name]
    datapoints_source.data = seg_dictionary

    cluster_select_table_source.data=OD
    
    th=list(set(OD['thr']))
    mapper['transform'].factors=th
    
    hmax,hedges,h_hist_df=hist_variables_create(OD[X])      

    hmax1,hedges1,h_hist_df1=hist_variables_create(OD[X][OD['thr']=='1'],hedges)            


    h_hist.data = h_hist_df
    
    h_hist1.data = h_hist_df1
    
    
    
    vmax,vedges,v_hist_df=hist_variables_create(OD[Y])

    vmax1,vedges1,v_hist_df1=hist_variables_create(OD[Y][OD['thr']=='1'],vedges)

    
    
    v_hist.data =v_hist_df
    
    v_hist1.data =v_hist_df1

    ph.y_range.start=min(-hmax,-hmax1)
    ph.y_range.end=max(hmax,hmax1)
    pv.x_range.start=min(-vmax,-vmax1)
    pv.x_range.end=max(vmax,vmax1)
    
    
    seg_plot.xaxis.axis_label = X
    seg_plot.yaxis.axis_label = Y
    

#report_data
observe_segment = Button(label='Observe Segment',button_type='success')
observe_segment.on_click(segment_func)


cluster_select_data=OD
cluster_select_table_Columns = [TableColumn(field=Ci, title=Ci) for Ci in cluster_select_data.columns] 
cluster_select_table_source=ColumnDataSource(cluster_select_data.round(decimals=2))
cluster_select_table = DataTable(columns=cluster_select_table_Columns, source=cluster_select_table_source,width=table_width,height=table_height)

download_button = Button(label="Download", button_type="success")
download_button.js_on_click(CustomJS(args=dict(source=cluster_select_table_source),
                            code=open(join(dirname(__file__), "download.js")).read()))

cluster_select_panel=Panel(child=column(row(cluster_select_table,width=table_width,height=table_height),download_button),title='Cluster Table')





















"""


END OF SEGMNENTATION


"""


#Data Description and Selected Data panel
tabs = Tabs(tabs=[dsc_panel,input_panel,sd_panel,output_panel,report_panel,cluster_select_panel],height=400)



layout=column(row(tabs),row(column(multi_select,row(select_algorithm,select_scaler),
              #row(arg_text),
              row(input_arguments,scaler_input_arguments),cluster_preset_checkbox,row(update_bt),reset_bt,width=500),column(hist_plot,hist_bt),column(seg_plot,observe_segment),column(Xselect,Yselect,Zselect,threhsold_slider)),joint_plot)

curdoc().add_root(layout)