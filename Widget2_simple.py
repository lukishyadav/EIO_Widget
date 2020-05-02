#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 16:44:08 2020
@author: lukishyadav
"""
from bokeh.models import TextInput,LinearColorMapper
from bokeh.io import curdoc
#import logging
from bokeh.layouts import column,layout,row,widgetbox
import pandas as pd
#import my_module
import datetime
#import seaborn as sns
#from pyproj import Proj
from bokeh.plotting import figure, show, output_file, ColumnDataSource
from bokeh.transform import factor_cmap
from bokeh.tile_providers import CARTODBPOSITRON 
import numpy as np
#from sklearn.cluster import DBSCAN 
from bokeh.models.widgets import Button, RadioButtonGroup, Select, Slider,TextInput,TextAreaInput
from bokeh.models import TextInput
from collections import Counter

from datetime import date
from bokeh.models.widgets import RadioButtonGroup
from bokeh.models.widgets import DateRangeSlider,DateSlider
from bokeh.models import RangeSlider

from bokeh.models.annotations import Title
from bokeh.models import Div
from bokeh.models import Panel, Tabs
from bokeh.models.widgets import DatePicker
from datetime import timedelta
from bokeh.models import CheckboxGroup
from bokeh.models import HoverTool,ResetTool
from bokeh.models import LinearColorMapper, BasicTicker, ColorBar,FixedTicker
from bokeh.tile_providers import CARTODBPOSITRON, get_provider,OSM,ESRI_IMAGERY
from bokeh.models import Panel, Tabs
from bokeh.models.widgets import DatePicker
from bokeh.models.renderers import  TileRenderer
from bokeh.models import Paragraph
map_repr='mercator'

dfile='generated_data/vehicles_data.csv'
dfile='generated_data/daytona_rental_datagenerated.csv'

#dfile='daytona_rental_data.csvgenerated.csv'
df=pd.read_csv(dfile)
df=df[df['within_radius']==1]
#df=df.sample(int(5*len(df)/100))

global Mindate
global Maxdate
Mindate=datetime.datetime.strptime(min(df['rental_ended_at_x'])[0:13], '%Y-%m-%d %H')
Maxdate=datetime.datetime.strptime(max(df['rental_ended_at_x'])[0:13], '%Y-%m-%d %H')


hour_values=(Mindate.hour,Maxdate.hour)

drs_start=DatePicker(title='Date',min_date=datetime.date(2017, 1, 1),max_date=datetime.date(2020, 6, 1),value=Mindate.date())
drs_start_hour = Select(title='Hour of the day',value='0', options=[str(x) for x in list(range(0,24))])

drs_end=DatePicker(title='Date',min_date=datetime.date(2017, 1, 1),max_date=datetime.date(2020, 6, 1),value=Mindate.date())
drs_end_hour = Select(title='Hour of the day',value='1', options=[str(x) for x in list(range(0,24))])



fdrs_start=DatePicker(title='Date',min_date=datetime.date(2017, 1, 1),max_date=datetime.date(2020, 6, 1),value=Mindate.date())
fdrs_start_hour = Select(title='Hour of the day',value='0', options=[str(x) for x in list(range(0,24))])

fdrs_end=DatePicker(title='Date',min_date=datetime.date(2017, 1, 1),max_date=datetime.date(2020, 6, 1),value=Mindate.date())
fdrs_end_hour = Select(title='Hour of the day',value='1', options=[str(x) for x in list(range(0,24))])


pdict={}

pdict['active']=3


year_values=(Mindate.year,Maxdate.year)
month_values=(Mindate.month,Maxdate.month)
day_values=(Mindate.day,Maxdate.day)
hour_values=(Mindate.hour,Maxdate.hour)

#display_columns=df.columns

#dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')


svalue="2020-01-03 00:00:00"
evalue="2020-01-03 01:00:00"

max_hours=24

vdict=dict()
vdict['hovertool_widget']=0



#df=df[(df['rental_ended_at_x']<=svalue) & (df['rental_started_at_y']>=evalue)]


d_start_date=Mindate+timedelta(hours=1)
d_end_date=Mindate+timedelta(hours=24)


df=df[(df['rental_ended_at_x']<str(d_end_date)) & (df['rental_started_at_y']>str(d_start_date))]






df['rental_ended_at_x']=df['rental_ended_at_x'].apply(lambda x:datetime.datetime.strptime(x[0:19], '%Y-%m-%d %H:%M:%S'))
df['rental_started_at_y']=df['rental_started_at_y'].apply(lambda x:datetime.datetime.strptime(x[0:19], '%Y-%m-%d %H:%M:%S'))
df['idle_hours']=df.apply(lambda x:(((x['rental_started_at_y']-x['rental_ended_at_x']).total_seconds())/3600),axis=1)

df=df[(df['idle_hours']<150) & (df['idle_hours']>0) ]
#df = pd.read_csv(infile, parse_dates=['rental_booked_at_y', 'rental_started_at_y', 'rental_ended_at_y','rental_booked_at_x', 'rental_started_at_x', 'rental_ended_at_x'], date_parser=dateparse)


display_columns=df.columns


#df = pd.read_csv(infile, parse_dates=['rental_ended_at_y','rental_started_at_x'], date_parser=dateparse)
datapoints_source = ColumnDataSource()

dictionary={}

global fd
fd=df.copy()
        
for col_name in display_columns:
# if col_name not in [X,Y]:
  dictionary[col_name]=df[col_name]

datapoints_source.data = dictionary  
 

maxlat=max(df['mrc_end_lat_x'])
minlat=min(df['mrc_end_lat_x'])

maxlng=max(df['mrc_end_long_x'])
minlng=min(df['mrc_end_long_x'])


run=0

#def my_slider_handler(attr,old,new):
def my_slider_handler(): 
    #try_text.text="Updating"
    
    print('start')
    pre.text='<h4><br><b style="color:slategray">Update in Progress....</b><br></h4>'
    completed.text=''
    #range_slider1=idle_range_slider
    
    #date_range_slider1=date_range_slider
    
    hour_max=hour_range_slider.value[1]
    hour_min=hour_range_slider.value[0]
    #print(type(vcheck[0]))
    #print(vcheck[0])
    #cc=datetime.datetime.fromtimestamp(vcheck[0]/1000).strftime('%Y-%m-%d %H:%M:%S')
    #print('Slider Check',cc)
    #print(vcheck[1])
    
    """
    SLIDER WIDGETS
    
    if isinstance(range_slider1.value[0], (int, float)):
    # pandas expects nanoseconds since epoch
        start_date = pd.Timestamp(float(range_slider1.value[0])*1e6)
        end_date = pd.Timestamp(float(range_slider1.value[1])*1e6)
    else:
        start_date = pd.Timestamp(range_slider1.value[0])
        end_date = pd.Timestamp(range_slider1.value[1])
        
    if isinstance(date_range_slider1.value[0], (int, float)):
    # pandas expects nanoseconds since epoch
        d_start_date = pd.Timestamp(float(date_range_slider1.value[0])*1e6)
        d_end_date = pd.Timestamp(float(date_range_slider1.value[1])*1e6)
    else:
        d_start_date = pd.Timestamp(date_range_slider1.value[0])
        d_end_date = pd.Timestamp(date_range_slider1.value[1])    
    
    """
    #print('dt_pckr_strt',dt_pckr_strt.value,type(dt_pckr_strt.value),dt_pckr_strt.value.year)
    #print(dre_year.value,dre_month.value,dre_day.value,dre_hour.value)
    #start_date=idle_range_start.value
    #end_date=idle_range_end.value
    
    
    
    
    #d_start_date=date_range_start.value
    #d_end_date=date_range_end.value
    
    
    #print(start_date,end_date)
    
    if radio_button_group.active==0:
        df=pd.read_csv('generated_data/darwin_rental_datagenerated.csv')
    elif radio_button_group.active==1:
        df=pd.read_csv('generated_data/darwin_e_rental_datagenerated.csv')
    elif radio_button_group.active==2:
        df=pd.read_csv('generated_data/darwin_s_rental_datagenerated.csv')
    elif radio_button_group.active==3:
        df=pd.read_csv('generated_data/daytona_rental_datagenerated.csv')
    else:    
        df=pd.read_csv('generated_data/eiffel_rental_datagenerated.csv')
        
    #df=pd.read_csv(dfile)
    df=df[df['within_radius']==1]
    
    
    """
    Mindate=datetime.datetime.strptime(min(df['rental_ended_at_x'])[0:13], '%Y-%m-%d %H')
    Mindate=datetime.datetime.strptime(min(df['rental_ended_at_x'])[0:19], '%Y-%m-%d %H:%M:%S')
    Maxdate=datetime.datetime.strptime(max(df['rental_ended_at_x'])[0:19], '%Y-%m-%d %H:%M:%S')
    
    
    
    year_values=(Mindate.year,Maxdate.year)
    month_values=(Mindate.month,Maxdate.month)
    day_values=(Mindate.day,Maxdate.day)
    hour_values=(Mindate.hour,Maxdate.hour)
    
    
    if radio_button_group.active!=pdict['active']:
        drs_start.value=Mindate
        drs_start_hour.value=str(hour_values[0])
        print('drs_start_hour',drs_start_hour.value)
        
        drs_end.value=Maxdate
        drs_end_hour.value=str(hour_values[1])
        print('drs_end_hour',drs_end_hour.value)
    
    """
    """
    d_start_date=Mindate+timedelta(hours=date_widget.value[0])
    d_end_date=Mindate+timedelta(hours=date_widget.value[1])
    
    print('d_start_date',d_start_date)
    print('d_end_date',d_end_date)
    
    start_date=Mindate+timedelta(hours=fine_date_widget.value[0])
    end_date=Mindate+timedelta(hours=fine_date_widget.value[1])
    
    print('start_date',start_date)
    print('end_date',end_date)
    
    print('Mindate',Mindate)
    print('radio buttomn active',radio_button_group.active)
    """
    #wid=drs_start.value
    wid=datetime.datetime.strptime(drs_start.value[0:10], '%Y-%m-%d').date()
    d_start_date=datetime.datetime(wid.year,wid.month,wid.day,int(drs_start_hour.value))
    print('d_start_date',d_start_date)
    
    wid2=datetime.datetime.strptime(drs_end.value[0:10], '%Y-%m-%d').date()
    d_end_date=datetime.datetime(wid2.year,wid2.month,wid2.day,int(drs_end_hour.value))
    print('d_end_date',d_end_date)


    fid=datetime.datetime.strptime(fdrs_start.value[0:10], '%Y-%m-%d').date()
    start_date=datetime.datetime(fid.year,fid.month,fid.day,int(fdrs_start_hour.value))
    print('start_date',start_date)
    
    fid2=datetime.datetime.strptime(fdrs_end.value[0:10], '%Y-%m-%d').date()
    end_date=datetime.datetime(fid2.year,fid2.month,fid2.day,int(fdrs_end_hour.value))
    print('end_date',end_date)
    
    #df=df.sample(int(5*len(df)/100))
    #svalue=sdate_input.value
    #print(pd.datetime.strptime(svalue, '%Y-%m-%d %H:%M:%S'))
    #evalue=edate_input.value
    #print(evalue)
    #df=df[(df['rental_ended_at_x']<evalue) & (df['rental_started_at_y']>svalue)]
    dictionary={}
    try:
        #print(len(df),d_start_date,d_end_date)
        #print(hour_range_slider.value[1])
        if 0 in checkbox_group.active:
         print('1st')
         df=df[(df['rental_ended_at_x']<str(d_end_date)) & (df['rental_started_at_y']>str(d_start_date))]
         print('post 1st')
        if 1 in checkbox_group.active:
         print('2nd')   
         df=df[(df['rental_ended_at_x']<=str(start_date)) & (df['rental_started_at_y']>=str(end_date))]
        
        
        df['rental_ended_at_x']=df['rental_ended_at_x'].apply(lambda x:datetime.datetime.strptime(x[0:19], '%Y-%m-%d %H:%M:%S'))
        df['rental_started_at_y']=df['rental_started_at_y'].apply(lambda x:datetime.datetime.strptime(x[0:19], '%Y-%m-%d %H:%M:%S'))

        df['idle_hours']=df.apply(lambda x:(((x['rental_started_at_y']-x['rental_ended_at_x']).total_seconds())/3600),axis=1)

        df=df[(df['idle_hours']<hour_max) & (df['idle_hours']>hour_min)]

        maxlat=max(df['mrc_end_lat_x'])
        minlat=min(df['mrc_end_lat_x'])
        
        maxlng=max(df['mrc_end_long_x'])
        minlng=min(df['mrc_end_long_x'])
        
        if maxlng==minlng or maxlat==minlat:
            raise Exception("Sorry, no numbers below zero")
        #print(df[['mrc_end_lat_x','mrc_end_long_x']].values)
        
        
        
        global run
        
        #if pdict['active']!=radio_button_group.active or run==0:
        if radio_button_group.active!=pdict['active'] or span_radio.active==1 or run==0:  
            map_figure.toolbar.active_tap=None
            map_figure.x_range.end=maxlng
            map_figure.x_range.start=minlng
            map_figure.y_range.end=maxlat
            map_figure.y_range.start=minlat
                
            
        
        run=run+1
    
        """

        """
        
        print(maxlat,maxlng,minlat,minlng)
        print('within radius Counter',len(df['within_radius']))
        #print(hour_range_slider.value[1])
        global fd
        fd=df.copy()
        display_columns=df.columns
        for col_name in display_columns:
        # if col_name not in [X,Y]:
          dictionary[col_name]=df[col_name]
        
        datapoints_source.data = dictionary  
        
        """
        circle_plot.glyph.size=size_range_slider.value
    
        circle_plot.glyph.fill_alpha=alpha_range_slider.value
        """
        
        if map_type.active==0:
            pre.text='<h4><br><b style="color:slategray">High Utilization Areas</b><br>'+'<b style="color:slategray">Filtered Idle spots: </b>'+str(len(df))+'</h4>'
            palet=decided_palet
            print('Uti palette',palet)
            #color_mapper = LinearColorMapper(palette=palet)
            color_mapper.palette=palet
            color_mapper.high=max(df['idle_hours'])
            color_mapper.low=min(df['idle_hours']) 
            color_bar.color_mapper=color_mapper
            circle_plot.glyph.fill_color={'field': 'idle_hours', 'transform': color_mapper}
        elif map_type.active==1:
            pre.text='<h4><br><b style="color:slategray">Aging Cars</b><br>'+'<b style="color:slategray">Filtered Idle spots: </b>'+str(len(df))+'</h4>'
            #print(map_figure.title.text)
            palet=decided_palet
            palet=palet[::-1]
            print('aging palette',palet)
            #color_mapper = LinearColorMapper(palette=palet)
            color_mapper.palette=palet
            color_mapper.high=max(df['idle_hours'])
            color_mapper.low=min(df['idle_hours']) 
            color_bar.color_mapper=color_mapper
            circle_plot.glyph.fill_color={'field': 'idle_hours', 'transform': color_mapper}


    except Exception as e:
        #pass
        print(e)
        
        
        fdd=fd.copy()
        fdd=fdd.head(1)
        display_columns=fdd.columns
        for col_name in display_columns:
        # if col_name not in [X,Y]:
          dictionary[col_name]=[]
        
        datapoints_source.data = dictionary 
        
        
        pre.text='<h4><br><b style="color:slategray">Error! Try expanding date slider</b><br></h4>'
        
        
        
        #print(dictionary)
    #pre.text= """Filtered Idle spots:"""+str(len(df))  

    point_selector.active=[0,1,2,3,4]
    #t.text="start"    
   
    
    pdict['active']= radio_button_group.active   
    completed.text='<b>Update Completed!</b>'


    
def update_click():
    pre.text='<h4><br><b style="color:slategray">Update in Progress....</b><br></h4>'
    curdoc().add_next_tick_callback(my_slider_handler)
    

sdate_input = TextInput(value="2020-01-03 00:00:00", title="Start Date: (YYYY-MM-DD HH:MM:SS)")
#sdate_input.on_change("value", my_slider_handler)

edate_input = TextInput(value="2020-01-03 01:00:00", title="End Date: (YYYY-MM-DD HH:MM:SS)")
#edate_input.on_change("value", my_slider_handler)



date_range_slider = DateRangeSlider(title="Data Filter Date Range: ", start=datetime.datetime(2020, 1, 3,0), end=datetime.datetime(2020, 3, 20,1), value=(datetime.datetime(2020, 1, 3,0), datetime.datetime(2020, 1, 3,1)),format="%x,%X")


date_range_start = TextInput(value="2020-01-03 00:00:00", title="Data Filter Start Date: (YYYY-MM-DD HH:MM:SS)")
#sdate_input.on_change("value", my_slider_handler)

date_range_end = TextInput(value="2020-01-03 01:00:00", title="Data Filter End Date: (YYYY-MM-DD HH:MM:SS)")





idle_range_slider = DateRangeSlider(title="Idle Date Range: ", start=datetime.datetime(2020, 1, 3,0), end=datetime.datetime(2020, 3, 20 ,1), value=(datetime.datetime(2020, 1, 3,0), datetime.datetime(2020, 1, 3,1)),format="%x,%X")


idle_range_start = TextInput(value="2020-01-03 00:00:00", title="Idle Filter Start Date: (YYYY-MM-DD HH:MM:SS)")
#sdate_input.on_change("value", my_slider_handler)

idle_range_end = TextInput(value="2020-01-03 01:00:00", title="Idle Filter End Date: (YYYY-MM-DD HH:MM:SS)")

#sdate_range_slider = DateRangeSlider(title="Date Range: ", start=datetime.datetime(2017, 1, 1,1), end=datetime.datetime(2017, 2, 7,2), value=(datetime.datetime(2017, 9, 7,1), datetime.datetime(2017, 9, 7,2)),step=1)
#sdate_range_slider.on_change("value", my_slider_handler)
#sdate_range_slider = DateSlider(title="Date Range: ", start=datetime.datetime(2017, 1, 1,1), end=datetime.datetime(2019, 9, 7,2), value=(datetime.datetime(2017, 9, 7,1), datetime.datetime(2017, 9, 7,2)), step=1)


date_range_radio = RadioButtonGroup(name='Date Range Filter',
        labels=["Date Range Filter On", "Date Range Filter Off"], active=0)


idle_range_radio= RadioButtonGroup(name='Idle  Range Filter',
        labels=["Idle  Range Filte On", "Idle  Range Filte Off"], active=1)


checkbox_group = CheckboxGroup(
        labels=["Cumulative Date Filter", "Fine Date Filter"], active=[0])



hour_range_slider = RangeSlider(start=0, end=360, value=(0,150), step=1, title="Idle time for a vehicle (hours)")

alpha_range_slider = Slider(start=0, end=1, value=0.4, step=.1, title="Spot Transparency")

size_range_slider = Slider(start=4, end=50, value=4, step=1, title="Spot Size")


def alpha_size(attr, old, new):
    circle_plot.glyph.size=size_range_slider.value
    
    circle_plot.glyph.fill_alpha=alpha_range_slider.value
    

alpha_range_slider.on_change('value', alpha_size)

size_range_slider.on_change('value', alpha_size)


bt = Button(label='Update Plot',default_size=300,css_classes=['custom_button_1'])
bt.on_click(update_click)


hovertool_widget = RadioButtonGroup(
        labels=["No Hover tool", "Hover tool"], active=0)

hovertool_timer=TextAreaInput(value="", rows=1, title="Select display time for tooltip")




radio_button_group = RadioButtonGroup(
        labels=["Darwin","Darwin E","Darwin S","Daytona","Eiffel"], active=3)



date_text = Div(text='<b style="color:black">'+str(Mindate)+'&nbsp;&nbsp;&nbsp;to&nbsp;&nbsp;&nbsp;'+str(Mindate+timedelta(hours=24))+'<br></b>',width=500, height=40)



fine_date_text = Div(text='<b style="color:black">'+str(Mindate)+'&nbsp;&nbsp;&nbsp;to&nbsp;&nbsp;&nbsp;'+str(Maxdate)+'<br></b>',width=500, height=40)


def date_function(attr, old, new):
    NMindate=Mindate+timedelta(hours=date_widget.value[0])
    NMaxdate=Mindate+timedelta(hours=date_widget.value[1])
    
    fNMindate=Mindate+timedelta(hours=fine_date_widget.value[0])
    fNMaxdate=Mindate+timedelta(hours=fine_date_widget.value[1])
    
    date_text.text='<b style="color:black">'+str(NMindate)+'&nbsp;&nbsp;&nbsp;to&nbsp;&nbsp;&nbsp;'+str(NMaxdate)+'<br></b>'
    
    fine_date_text.text='<b style="color:black">'+str(fNMindate)+'&nbsp;&nbsp;&nbsp;to&nbsp;&nbsp;&nbsp;'+str(fNMaxdate)+'<br></b>'

date_widget = RangeSlider(start=0, end=4400, value=(1,24), step=1,show_value=False,tooltips=False)

date_widget.on_change('value', date_function)


fine_date_widget = RangeSlider(start=0, end=4400, value=(1,24), step=1,show_value=False,tooltips=False)

fine_date_widget.on_change('value', date_function)
  

def point_selector_f(attr, old, new):
    
    if map_type.active==0:
        print(max(fd['idle_hours']))
        max_idle=max(fd['idle_hours'])
        min_idle=min(fd['idle_hours'])
        idle_interval=(max_idle-min_idle)/5
        print(idle_interval)
        if 0 in point_selector.active:
            fd1=fd[fd['idle_hours']<idle_interval]
        else:
            fd1=pd.DataFrame(data=[],columns=[])
        if 1 in point_selector.active:
            fd2=fd[(fd['idle_hours']>=idle_interval) & (fd['idle_hours']<2*idle_interval)]
        else:
            fd2=pd.DataFrame(data=[],columns=[])
        if 2 in point_selector.active:
            fd3=fd[(fd['idle_hours']>=2*idle_interval) & (fd['idle_hours']<3*idle_interval)]
        else:
            fd3=pd.DataFrame(data=[],columns=[])
        if 3 in point_selector.active:
            fd4=fd[(fd['idle_hours']>=3*idle_interval) & (fd['idle_hours']<4*idle_interval)]
        else:
            fd4=pd.DataFrame(data=[],columns=[])
        if 4 in point_selector.active:
            fd5=fd[(fd['idle_hours']>=4*idle_interval)]
        else:
            fd5=pd.DataFrame(data=[],columns=[])
    else:   
        #print(max(fd['idle_hours']))
        max_idle=max(fd['idle_hours'])
        min_idle=min(fd['idle_hours'])
        idle_interval=(max_idle-min_idle)/5
        print(idle_interval)
        if 0 in point_selector.active:
            fd1=fd[fd['idle_hours']>=4*idle_interval]
        else:
            fd1=pd.DataFrame(data=[],columns=[])
        if 1 in point_selector.active:
            fd2=fd[(fd['idle_hours']>=3*idle_interval) & (fd['idle_hours']<4*idle_interval)]
        else:
            fd2=pd.DataFrame(data=[],columns=[])
        if 2 in point_selector.active:
            fd3=fd[(fd['idle_hours']>=2*idle_interval) & (fd['idle_hours']<3*idle_interval)]
        else:
            fd3=pd.DataFrame(data=[],columns=[])
        if 3 in point_selector.active:
            fd4=fd[(fd['idle_hours']>=idle_interval) & (fd['idle_hours']<2*idle_interval)]
        else:
            fd4=pd.DataFrame(data=[],columns=[])
        if 4 in point_selector.active:
            fd5=fd[(fd['idle_hours']<idle_interval)]
        else:
            fd5=pd.DataFrame(data=[],columns=[])
    
    """    
    if 5 in point_selector.active:
        pass
    else:
        fd2=pd.DataFrame(data=[],columns=[])
    """      
    df_main_list=[]  
    df_list=[fd1,fd2,fd3,fd4,fd5]
    for x in df_list:
       if len(x)!=0:
           df_main_list.append(x)
    
    try:  
        FD=pd.concat(df_main_list, ignore_index=True)       
    

        display_columns=FD.columns
        for col_name in display_columns:
        # if col_name not in [X,Y]:
          dictionary[col_name]=FD[col_name]
        
        datapoints_source.data = dictionary
        
        if map_type.active==0:
            pre.text='<h4><br><b style="color:slategray">High Utilization Areas</b><br>'+'<b style="color:slategray">Filtered Idle spots: </b>'+str(len(FD))+'</h4>'
        elif map_type.active==1:
            pre.text='<h4><br><b style="color:slategray">Aging Cars</b><br>'+'<b style="color:slategray">Filtered Idle spots: </b>'+str(len(FD))+'</h4>'

    except:
        display_columns=fd.columns
        for col_name in display_columns:
        # if col_name not in [X,Y]:
          dictionary[col_name]=[]
        
        datapoints_source.data = dictionary 
        pre.text='<h4><br><b style="color:slategray">Error! Try expanding date slider</b><br></h4>'
        
        

point_selector = CheckboxGroup(
        labels=["", "","","",""], active=[0,1,2,3,4])

point_selector.on_change('active', point_selector_f)


def drs_function():
    #pre.text='<h4 style="border-top: 2px solid #778899;width: 1600px"><br><b style="color:slategray">Adjusting presets to reduce load....</b><br></h4>'
    if radio_button_group.active==0:
        df=pd.read_csv('generated_data/darwin_rental_datagenerated.csv')
    elif radio_button_group.active==1:
        df=pd.read_csv('generated_data/darwin_e_rental_datagenerated.csv')
    elif radio_button_group.active==2:
        df=pd.read_csv('generated_data/darwin_s_rental_datagenerated.csv')
    elif radio_button_group.active==3:
        df=pd.read_csv('generated_data/daytona_rental_datagenerated.csv')
    else:    
        df=pd.read_csv('generated_data/eiffel_rental_datagenerated.csv')
        
    #df=pd.read_csv(dfile)
    df=df[df['within_radius']==1]
    
    Mindate=datetime.datetime.strptime(min(df['rental_ended_at_x'])[0:19], '%Y-%m-%d %H:%M:%S')
    

    Maxdate=datetime.datetime.strptime(max(df['rental_ended_at_x'])[0:19], '%Y-%m-%d %H:%M:%S')
    
    #print('min,max',Mindate,Maxdate)
    hour_values=(Mindate.hour,Maxdate.hour)

    
    drs_start.min_date=Mindate.date()
    drs_start.max_date=Maxdate.date()
    drs_start_hour.value=str(hour_values[0])
    drs_start.value=Mindate.date()
    # print('drs_start_hour',drs_start_hour.value)
    
    
    drs_end.min_date=Mindate.date()
    drs_end.max_date=Maxdate.date()
    drs_end_hour.value=str((Mindate+ timedelta(hours=24)).hour)
    drs_end.value=(Mindate+ timedelta(hours=24)).date()
    #print('drs_end_hour',drs_end_hour.value)
    
    
    fdrs_start.min_date=Mindate.date()
    fdrs_start.max_date=Maxdate.date()
    fdrs_start_hour.value=str(hour_values[0])
    fdrs_start.value=Mindate.date()
    #print('drs_start_hour',drs_start_hour.value)
    
    
    fdrs_end.min_date=Mindate.date()
    fdrs_end.max_date=Maxdate.date()
    fdrs_end_hour.value=str((Mindate+ timedelta(hours=24)).hour)
    fdrs_end.value=(Mindate+ timedelta(hours=24)).date()
    #print('drs_end_hour',drs_end_hour.value)
    
    pre.text='<h4"><br><br><b style="color:slategray">You can start adjusting the widgets now!</b><br></h4>'
 
    
def update_radio_button_group(attr, old, new):
    pre.text='<h4><br><b style="color:slategray">Adjusting presets to reduce load....</b><br></h4>'
    curdoc().add_next_tick_callback(drs_function)

    
radio_button_group.on_change('active',update_radio_button_group)


map_type=RadioButtonGroup(
        labels=["Find High Utilization Areas","Find Aging Cars"], active=0)


span_radio=RadioButtonGroup(
        labels=["Lock Map Area","Auto Adjust Map Area"], active=0)


weekday_checkbox=CheckboxGroup(labels=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'],active=[0,1,2,3,4,5,6])


TOOLTIP=HoverTool()
TOOLTIP_list=['<b style="color:MediumSeaGreen;">'+name_cols+':'+'</b><b>'+' @{'+name_cols+'}</b>' for name_cols in display_columns]
#TOOLTIP=[(name_cols,'@{'+name_cols+'}') for name_cols in display_columns]
TOOLTIP_end = "<br>".join(TOOLTIP_list)

TOOLTIP.tooltips= """
    <style>
        .bk-tooltip>div:not(:first-child) {display:none;}
    </style>"""+TOOLTIP_end



#add a dot where the click happened
def hovercallback(attr, old, new):
 if vdict['hovertool_widget']!=  hovertool_widget.active:  
    if hovertool_widget.active==1:
        map_figure.add_tools(TOOLTIP)
        print(map_figure.tools[-1])
    elif hovertool_widget.active==0:
        del map_figure.tools[-1]
#        p.add_tools(TOOLTIP)
    #print(p.tools)
    
#        time.sleep(float(hovertool_timer.value))
#        del p.tools[-1]
        
    #print(p.tools)
    

 vdict['hovertool_widget']= hovertool_widget.active
    

hovertool_widget.on_change('active',hovercallback)




# set up/draw the map
map_figure = figure(
    x_range=(minlng,maxlng),
    y_range=(minlat, maxlat),
    x_axis_type=map_repr,
    y_axis_type=map_repr,
#    title='High Utilisation Areas'
)


#map_figure.add_tile(CARTODBPOSITRON)

tile_provider = get_provider(CARTODBPOSITRON)
map_figure.add_tile(tile_provider)


tiles = {'Light':get_provider(CARTODBPOSITRON),'Open Street':get_provider(OSM),'Satellite':get_provider(ESRI_IMAGERY)}
#select menu

#callback
def change_tiles_callback(attr, old, new):
    #removing the renderer corresponding to the tile layer
    map_figure.renderers = [x for x in map_figure.renderers if not str(x).startswith('TileRenderer')]
    #inserting the new tile renderer
    tile_renderer = TileRenderer(tile_source=tiles[new])
    map_figure.renderers.insert(0, tile_renderer)
 

tile_prov_select = Select(title="Tile Provider", value="Light", options=['Light','Open Street','Satellite'])

    
tile_prov_select.on_change('value',change_tiles_callback) 


#show(map_figure)
  

from bokeh.palettes import Oranges,OrRd,RdYlGn,Reds

decided_palet=Reds[5]
palet=decided_palet
#palet.reverse()

color_mapper = LinearColorMapper(palette=palet)

color_mapper.high=max(df['idle_hours'])
color_mapper.low=min(df['idle_hours'])


"""
    map_figure.circle(x='mrc_end_long_x', y='mrc_end_lat_x', 
                      #size=cluster_point_size,
                      fill_alpha=0.4,
                      source=datapoints_source,color="firebrick",
                      line_alpha=0)
"""

def plot_points(map_figure,datapoints_source):
    noise_point_size = 1
    cluster_point_size = 10

    
    map_figure.circle(x='mrc_end_long_x', y='mrc_end_lat_x', 
                      #size=cluster_point_size,
                      fill_alpha=0.4,
                      source=datapoints_source,fill_color={'field': 'idle_hours', 'transform': color_mapper},
                      line_alpha=0)


#plot_points(map_figure,datapoints_source)
noise_point_size = 1
cluster_point_size = 10

print(maxlat,maxlng,minlat,minlng)
circle_plot=map_figure.circle(x='mrc_end_long_x', y='mrc_end_lat_x', 
                  #size=cluster_point_size,
                  fill_alpha=0.4,
                  source=datapoints_source,fill_color={'field': 'idle_hours', 'transform': color_mapper},
                  line_alpha=0)



#baseline=float((color_mapper.high-color_mapper.low)/5)
#tick_list=[baseline,2*baseline,3*baseline,4*baseline]
color_bar = ColorBar(color_mapper=color_mapper, ticker= BasicTicker(),
                     location=(0,0))

#color_bar.click_policy="hide"

map_figure.add_layout(color_bar, 'right')


#map_figure.toolbar.active_tap=None

from bokeh.models import PreText

output_file("div.html")




pre = Div(text='<h4><br><b style="color:slategray">High Utilization Areas</b><br>'+'<b style="color:slategray">Filtered Idle spots: </b>'+str(len(df))+'</h4>')

completed = Div(text='<h4><b style="color:slategray">Update Completed!<br></b><h4>',
width=500, height=10)

exception = Div(text='<b></b>',
width=500, height=20)

pstring=''
for p in range(len(palet)):
    pstring=pstring+'<b style="color:'+palet[p]+'">&#9606;</b><br>'

point_selector_text = Div(text=pstring,
width=500, height=20)


try_text = Paragraph(text="""start""")

carsharing_text = Div(text='<h2 style="border-bottom: 2px solid #778899;width: 1600px;color:darkslategray;font-family: "Lucida Console", Courier, monospace;">Car Sharing Utilization Tracking Tool<br><br></h2>',
width=500, height=40)
"""
dre_p1=Panel(child=row(drs_start,width=150), title="Date Start Filter")
dre_p12=Panel(child=row(drs_start_hour,width=130), title="Hour Start Filter")
dre_p1=Tabs(tabs=[dre_p1])
dre_p12=Tabs(tabs=[dre_p12])
dre_p2=Panel(child=row(drs_end,width=150), title="Date End Filter")
dre_p22=Panel(child=row(drs_end_hour,width=130), title="Hour End Filter")
dre_p2=Tabs(tabs=[dre_p2])
dre_p22=Tabs(tabs=[dre_p22])
"""



"""
dre_p3=Panel(child=row(dre_year,dre_month,dre_day,dre_hour,width=260), title="DRE_End")
dre_p3=Tabs(tabs=[dre_p3])
"""

CDF=Panel(child=column(widgetbox(date_text,height=30),widgetbox(date_widget,height=40,width=800)), title="Cumulative Date Filter")
FDF=Panel(child=column(widgetbox(fine_date_text,height=30),widgetbox(fine_date_widget,height=40,width=800)), title="Fine Date Filter")

date_tabs=Tabs(tabs=[CDF,FDF])



fdre_p1=Panel(child=row(fdrs_start,fdrs_start_hour,width=280), title="From")
fdre_p1=Tabs(tabs=[fdre_p1])
fdre_p2=Panel(child=row(fdrs_end,fdrs_end_hour,width=280), title="To")
fdre_p2=Tabs(tabs=[fdre_p2])
fine_df=Panel(child=row(column(row(fdre_p1,width=350)),column(row(fdre_p2,width=350))),title="Fine Date Range Filter")
#fine_df=Tabs(tabs=[fine_df])

dre_p1=Panel(child=row(drs_start,drs_start_hour,width=280), title="From")
dre_p1=Tabs(tabs=[dre_p1])
dre_p2=Panel(child=row(drs_end,drs_end_hour,width=280), title="To")
dre_p2=Tabs(tabs=[dre_p2])
cum_df=Panel(child=row(column(row(dre_p1,width=350)),column(row(dre_p2,width=350))),title="Cumulative Date Range Filter")
cum_df=Tabs(tabs=[cum_df,fine_df])



widgets=Panel(child=column(
                widgetbox(height=10),
                widgetbox(radio_button_group,height=50),  
                widgetbox(height=10),
                widgetbox(checkbox_group,height=50),
                widgetbox(height=10),
                widgetbox(hour_range_slider,height=50),
                widgetbox(height=10),
                widgetbox(map_type,height=50),
                widgetbox(height=10),
                date_tabs,
                cum_df,
                widgetbox(height=10),
                widgetbox(span_radio),
                widgetbox(height=10),
                widgetbox(bt,height=50),
                widgetbox(height=10),
                widgetbox(hovertool_widget),
                widgetbox(height=10),
                width=400)  , title="Base Widgets")


plot_options=Panel(child=column(widgetbox(tile_prov_select),
                widgetbox(alpha_range_slider),
                widgetbox(size_range_slider),
                row(widgetbox(point_selector_text,width=15),widgetbox(point_selector)),), title="Visibility Widgets")

w_tabs=Tabs(tabs=[widgets,plot_options])
layout = column(carsharing_text,row(column(row(pre,height=100),map_figure,
                width=800),column(widgetbox(height=100),row(w_tabs))
            
                #widgetbox(slider,width=350),
                #widgetbox(Min_n, width=300),
                #Percent,            
        ), 
        #row(widgetbox(date_range_slider,width=1400),width=2300),
        #row(widgetbox(idle_range_slider,width=1400),width=2300),
        #row(date_range_start,date_range_end),
        #row(idle_range_start,idle_range_end),
        #row(dre_p3,width=200),
        #row(drs_start,drs_end,width=400)
        
        )

curdoc().add_root(layout)
curdoc().title = 'EOIs'