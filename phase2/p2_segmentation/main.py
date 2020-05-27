#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 12:54:15 2020

@author: lukishyadav
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 14:03:19 2020

@author: lukishyadav
"""

from bokeh.io import curdoc
#import logging
from bokeh.layouts import column,layout,row,widgetbox
import pandas as pd
import datetime
from bokeh.plotting import figure, show, output_file, ColumnDataSource
from bokeh.transform import factor_cmap
from bokeh.tile_providers import CARTODBPOSITRON
import numpy as np
from bokeh.models.widgets import Button, RadioButtonGroup, Select, Slider,TextInput


import pandas as pd
import datetime
from numpy import mean
import collections
from collections import Counter as c



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

from bokeh.layouts import gridplot

sub=''

df = pd.read_csv(sub+'Darwin_the_data.csv')

#df.isnull().sum()

dsc=df.describe()

Z='Rental_Counts'
df['threshold']=df[Z].apply(lambda x:1 if x>1 else 0)

df['threshold']=df['threshold'].astype('str')


ldict=dict(c(df['threshold']))


#df['legend']=df['threshold'].apply(lambda x:x+': '+str(ldict[x]))


#df['Monetization']=df['Monetization'].astype('str')

dsc=df.describe()

CL=list(df.columns)
CL.remove('Customer_id')
CL.remove('threshold')
CL.remove('weekday')
#CL.remove('legend')

#datapoints_source = df

X='Age'
Y='Average_time_between_rental(including park time)'


vdict=dict()
vdict['Xold']=X
vdict['Yold']=Y
vdict['Zold']=Z
vdict['hovertool_widget']=0


"""
datapoints_source = ColumnDataSource()
datapoints_source.data = dict(
    x=df[X],
    y=df[Y],
    thr=df['legend'],
    cid=df['Customer_id']
    #time=datapoints_df['start_datetime']
    )

"""



display_columns=['Customer_id', 'Total_count_of_rentals_from_0-4(local time)',
       'Total_count_of_rentals_from_4-8(local time)',
       'Total_count_of_rentals_from_8-12(local time)',
       'Total_count_of_rentals_from_12-16(local time)',
       'Total_count_of_rentals_from_16-20(local time)',
       'Total_count_of_rentals_from_20-24(local time)','Total_fare(Total of all bills)',
       'Days_from_last_rental',
       'Avg_rental_duration (excluding park time)',
       'Average_time_between_rental(including park time)',
       'Average_bill_per_trip', 'Total_Coupon_amt_used',
       'Rental_Counts','Gender','Weekday_rental_counts', 'Weekend_rental_counts',
       'Monetization(range:1-4)', 'Recency(range:1-4)', 'Frequency(range:1-4)', 'Age','threshold',
       'No_of_paid_rides (Rides with (Fare - Coupon value) > 0)']

display_columns=df.columns

samp=1
dfs=df.sample(int(samp*len(df)/100),random_state=42)


dictionary=dict(x=dfs[X],
    y=dfs[Y],
    thr=dfs['threshold'])

for col_name in display_columns:
# if col_name not in [X,Y]:
  dictionary[col_name]=dfs[col_name]

datapoints_source = ColumnDataSource()
datapoints_source.data = dictionary



dpal=ColumnDataSource()
dpal.data = dict(
    th=list(set(df['Rental_Counts']))
    #time=datapoints_df['start_datetime']
    )


from bokeh.models.annotations import Title

import numpy as np

def my_slider_handler():
    #slider.value=new

    X=Xselect.value
    Y=Yselect.value
    Z=Zselect.value
    th=slider.value
    samp=sampling.value

    rs=int(text_input.value)





    print(radio_button_group.active)
    if radio_button_group.active==0:
        df=pd.read_csv(sub+'Darwin_the_data.csv')
    elif radio_button_group.active==1:
        df=pd.read_csv(sub+'DarwinEastBay_the_data.csv')
    elif radio_button_group.active==2:
        df=pd.read_csv(sub+'DarwinSacramento_the_data.csv')
    elif radio_button_group.active==3:
        df=pd.read_csv(sub+'Daytona_the_data.csv')
    else:    
        df=pd.read_csv(sub+'Eiffel_the_data.csv')
    X_sampling.start=min(df[X])
    X_sampling.end=max(df[X])

    Y_sampling.start=min(df[Y])
    Y_sampling.end=max(df[Y])

    if vdict['Xold']!=X:
     X_sampling.value=(min(df[X]),max(df[X]))
    if vdict['Yold']!=Y:
     Y_sampling.value=(min(df[Y]),max(df[Y]))
    xmin=X_sampling.value[0]
    xmax=X_sampling.value[1]
    ymin=Y_sampling.value[0]
    ymax=Y_sampling.value[1]

    #   source.change.emit()
    if np.any([isinstance(val, str) for val in df[Z]]):
        df['threshold']=df[Z]



        filtered_df=df



    else:
        filtered_df=df
        df['threshold']=df[Z].apply(lambda x:1 if x>th else 0)
        df['threshold']=df['threshold'].astype('str')



        slider.start=min(df[Z])
        slider.end=max(df[Z])
        c_sampling.start=min(df[Z])
        c_sampling.end=max(df[Z])
        filtered_df=df
        if vdict['Zold']!=Z:
         c_sampling.value=(min(df[Z]),max(df[Z]))
        cmin=c_sampling.value[0]
        cmax=c_sampling.value[1]
        filtered_df=filtered_df[(filtered_df[Z]>=cmin) & (filtered_df[Z]<=cmax)]
        #th=['0','1']






    filtered_df=filtered_df[(filtered_df[X]>=xmin) & (filtered_df[X]<=xmax)]
    filtered_df=filtered_df[(filtered_df[Y]>=ymin) & (filtered_df[Y]<=ymax)]


    filtered_df=filtered_df.sample(int(samp*len(filtered_df)/100),random_state=rs)

    #ldict=dict(c(filtered_df['threshold']))
    #filtered_df['legend']=filtered_df['threshold'].apply(lambda x:x+': '+str(ldict[x]))

    th=list(set(filtered_df['threshold']))
    mapper['transform'].factors=th
    #noise_source = ColumnDataSource()
    #datapoints_source = ColumnDataSource()

    # modify columndatasources to modify the figures
    """
    datapoints_source.data = dict(
        x=filtered_df[X],
    y=filtered_df[Y],
    thr=filtered_df['legend'],
    cid=filtered_df['Customer_id']
        #time=noise_df['start_datetime']
        )
    """

    dictionary=dict(x=filtered_df[X],
    y=filtered_df[Y],
    thr=filtered_df['threshold'])

    for col_name in display_columns:
    # if col_name not in [X,Y]:
      dictionary[col_name]=filtered_df[col_name]


    datapoints_source.data = dictionary


    ldict=dict(c(filtered_df['threshold']))
    ldictkeys=list(ldict.keys())
    ldictvalues=list(ldict.values())
    ldictlist=['<b>'+str(ldictkeys[f])+'</b>'+': '+str(ldictvalues[f]) for f in range(len(ldictvalues))]
    ldictfinal='\n'.join(ldictlist)



    print(ldictfinal)
    p.xaxis.axis_label = X
    p.yaxis.axis_label = Y
    legend_title = Z+"_Threshold"
    plot_text.text=text='<b style="color:Gray;"></b><br><b style="color:MediumSeaGreen;">Points displayed : Total points in the dataset: </b>'+str(len(filtered_df))+':'+str(len(df))+'\n'+'<b style="color:MediumSeaGreen;"><br>Counts relative to Threshold: </b>'+ldictfinal




    vdict['Xold']=X
    vdict['Yold']=Y
    vdict['Zold']=Z

bt = Button(label='Update Plot')
bt.on_click(my_slider_handler)

from bokeh.plotting import figure, show
#from bokeh.sampledata.iris import flowers as df
from bokeh.transform import factor_cmap

from bokeh.models.widgets import Slider
from bokeh.io import output_file, show
from bokeh.models.widgets import RangeSlider
from bokeh.models.widgets import Select





global slider

slider = Slider(start=min(df[Z]), end=max(df[Z]), value=1, step=1, title="Color Threshold (For determining color of points)")

#slider.on_change('value', my_slider_handler)

global sampling

sampling = Slider(start=0, end=100, value=1, step=1, title="% Random Sampling to reduce over plotting.")

#sampling.on_change('value', my_slider_handler)


global c_sampling

c_sampling = RangeSlider(start=min(df[Z]), end=max((df[Z])), value=(min(df[Z]),max(df[Z])), step=1, title="Color Threshold Column Range")

#c_sampling.on_change('value', my_slider_handler)

global X_sampling

X_sampling = RangeSlider(start=min(df[X]), end=max((df[X])), value=(min(df[X]),max(df[X])), step=1, title="X axis range")

#X_sampling.on_change('value', slide_change)


global y_sampling

Y_sampling = RangeSlider(start=min(df[Y]), end=max(df[Y]), value=(min(df[Y]),max(df[Y])), step=1, title="Y axis range")

#Y_sampling.on_change('value', slide_change)

preset_sampling = Slider(start=0, end=100, value=10, step=1, title="% Preset Sampling.")




Xselect = Select(title="X-axis", value="Age", options=CL)
#Xselect.on_change('value', my_slider_handler)

Yselect = Select(title="Y-axis", value="Average_time_between_rental(including park time)", options=CL)
#Yselect.on_change('value', my_slider_handler)

Zselect = Select(title="Color Threshold Column", value="Rental_Counts", options=CL)
#Zselect.on_change('value', my_slider_handler)


from bokeh.io import output_file, show
from bokeh.models.widgets import TextAreaInput

text_input = TextAreaInput(value="42", rows=1, title="Random State (For repeatable sampling):")
#text_input.on_change('value', my_slider_handler)

#TOOLTIP=[('threshold', '@thr'),('Customer_id', '@cid')]



#TOOLTIP=[(name_cols,'@{'+name_cols+'}') for name_cols in display_columns]
#TOOLTIP = [TOOLTIP[0]]


from bokeh.models import HoverTool
TOOLTIP=HoverTool()
TOOLTIP_list=['<b style="color:MediumSeaGreen;">'+name_cols+':'+'</b><b>'+' @{'+name_cols+'}</b>' for name_cols in display_columns]
#TOOLTIP=[(name_cols,'@{'+name_cols+'}') for name_cols in display_columns]
TOOLTIP_end = "<br>".join(TOOLTIP_list)

TOOLTIP.tooltips= """
    <style>
        .bk-tooltip>div:not(:first-child) {display:none;}
    </style>"""+TOOLTIP_end



from bokeh.models.widgets import RadioButtonGroup

radio_button_group = RadioButtonGroup(
        labels=["Darwin","Darwin East Bay","Darwin Sacramento","Daytona","Eiffel"], active=0)




def g_preset_handler(sv1_values,sv2_values):
    
 slider_variables1=[slider,sampling,c_sampling,X_sampling,Y_sampling,Xselect,Yselect,Zselect,text_input ]
 slider_variables2=[radio_button_group]

 for wid in range(len(slider_variables1)):
     slider_variables1[wid].value=sv1_values[wid]

 for wid in range(len(slider_variables2)):
     slider_variables2[wid].active=sv2_values[wid]

 my_slider_handler() 
 
 
def my_slider_handler1():

 sv1_values=[6,preset_sampling.value,(0,100),(0,120),(0,2000),'Days_from_last_rental','Total_revenue(Total of actual fees paid)','Avg_rental_duration (including park time)','42']
 sv2_values=[0,1]
 g_preset_handler(sv1_values,sv2_values)
 

dpreset1 = Button(label='Darwin - 90 day churn analysis')
dpreset1.on_click(my_slider_handler1)



def my_slider_handler2():
     sv1_values=[250,preset_sampling.value,(0,2000),(0,120),(0,1500),'No_of_paid_rides (Rides with (Fare - Coupon value) > 0)','Average_time_between_rental(Excluding park time)','Total_revenue(Total of actual fees paid)','42']
     sv2_values=[0,0]
     g_preset_handler(sv1_values,sv2_values)

dpreset2 = Button(label='Darwin - LTV')
dpreset2.on_click(my_slider_handler2) 
 



def my_slider_handler3():
     sv1_values=[2,preset_sampling.value,(0,0),(0,83),(0,9727),'Age','Average_time_between_rental(including park time)','Gender','42']
     sv2_values=[0,0]
     g_preset_handler(sv1_values,sv2_values)
     
dpreset3 = Button(label='Darwin - Demographics')
dpreset3.on_click(my_slider_handler3) 


def my_slider_handler4():
     sv1_values=[20,preset_sampling.value,(0,90),(0,5264),(0,9269),'Total_Coupon_amt_used','No_of_paid_rides (Rides with (Fare - Coupon value) > 0)','Days_from_last_rental','42']
     sv2_values=[0,0]
     g_preset_handler(sv1_values,sv2_values)

dpreset4 = Button(label='Darwin - Coupon usage')
dpreset4.on_click(my_slider_handler4) 


def my_slider_handler5():
     sv1_values=[25,preset_sampling.value,(18,83),(0,218),(0,177),'Total_count_of_rentals_from_20-24(local time)','Saturday_rental_counts','Age','42']
     sv2_values=[0,0]
     g_preset_handler(sv1_values,sv2_values)

dpreset5 = Button(label='Darwin - Use patterns')
dpreset5.on_click(my_slider_handler5) 


def my_slider_handler6():
     sv1_values=[0,preset_sampling.value,(0,0),(0,149),(0,24890),'Avg_rental_duration (excluding park time)','Total_revenue(Total of actual fees paid)','Gender','42']
     sv2_values=[0,0]
     g_preset_handler(sv1_values,sv2_values)

dpreset6 = Button(label='Darwin - Power Users')
dpreset6.on_click(my_slider_handler6) 





def emy_slider_handler1():

 sv1_values=[6,preset_sampling.value,(0,100),(0,120),(0,2000),'Days_from_last_rental','Total_revenue(Total of actual fees paid)','Avg_rental_duration (including park time)','42']
 sv2_values=[4,1]
 g_preset_handler(sv1_values,sv2_values)
 

epreset1 = Button(label='Eiffel - 90 day churn analysis')
epreset1.on_click(emy_slider_handler1)



def emy_slider_handler2():
     sv1_values=[250,preset_sampling.value,(0,2000),(0,120),(0,1500),'No_of_paid_rides (Rides with (Fare - Coupon value) > 0)','Average_time_between_rental(Excluding park time)','Total_revenue(Total of actual fees paid)','42']
     sv2_values=[4,0]
     g_preset_handler(sv1_values,sv2_values)

epreset2 = Button(label='Eiffel - LTV')
epreset2.on_click(emy_slider_handler2) 
 



def emy_slider_handler3():
     sv1_values=[2,preset_sampling.value,(0,0),(0,708),(0,9727),'No_of_paid_rides (Rides with (Fare - Coupon value) > 0)','Average_time_between_rental(including park time)','Gender','42']
     sv2_values=[4,0]
     g_preset_handler(sv1_values,sv2_values)
     
epreset3 = Button(label='Eiffel - Demographics')
epreset3.on_click(emy_slider_handler3) 


def emy_slider_handler4():
     sv1_values=[20,preset_sampling.value,(0,90),(0,5264),(0,9269),'Total_Coupon_amt_used','No_of_paid_rides (Rides with (Fare - Coupon value) > 0)','Days_from_last_rental','42']
     sv2_values=[4,0]
     g_preset_handler(sv1_values,sv2_values)

epreset4 = Button(label='Eiffel - Coupon usage')
epreset4.on_click(emy_slider_handler4) 


def emy_slider_handler5():
     sv1_values=[25,preset_sampling.value,(18,83),(0,218),(0,177),'Total_count_of_rentals_from_20-24(local time)','Saturday_rental_counts','Gender','42']
     sv2_values=[4,0]
     g_preset_handler(sv1_values,sv2_values)

epreset5 = Button(label='Eiffel - Use patterns')
epreset5.on_click(emy_slider_handler5) 


def emy_slider_handler6():
     sv1_values=[0,preset_sampling.value,(0,0),(0,149),(0,24890),'Avg_rental_duration (excluding park time)','Total_revenue(Total of actual fees paid)','Gender','42']
     sv2_values=[4,0]
     g_preset_handler(sv1_values,sv2_values)

epreset6 = Button(label='Eiffel - Power Users')
epreset6.on_click(emy_slider_handler6) 








style_dict1={'font-family': "Times New Roman",
  'font-style': 'bold',
#  'color': 'white',
  'border': '1px solid MediumSeaGreen',
  'border-radius': '25px',
  'border-style': 'ridge',
#  'background': '#73AD21',
  'padding': '20px',
  'width': '650px',
  'height': '100px',
}

style_dict2={'font-family': "Georgia, serif",
#  'font-style': 'bold',
  'border': '2px solid LightGray',
  'border-radius': '25px',
   'border-style': 'outset',
  'padding': '5px',
  'width': '525px',
  'height': '470px',
}



style_dict4={'font-family': "Times New Roman",'font-size': '20px',
  'font-style': 'bold',
  'color': 'crimson',
#  'border': '4px solid #73AD21',
#  'border-radius': '25px',
#  'background': '#73AD21',
#  'padding': '20px',
#  'width': '1050px',
#  'height': '110px',
}

style_dict5={'font-family': "Arnoldboecklin, fantasy",'font-size': '20px',
  'font-style': 'bold'}

style_dict6={'font-family': "Arnoldboecklin, fantasy",'font-size': '20px',
  'font-style': 'bold','color':'#73AD21'}






pre = PreText(text="""* Widgets to be used to alter the plot.You can:
 1. Select X ,Y-axis and the range of the same.
 2. Select feature to decide color of the points(Threshold)
    as well as  the value which decides the color.(> logic)   
 3. Select the range of threshold column to filter further. 
 4. Alter the amount of samples you see on plot.
  (CAUTION! MORE SAMPLES WILL LEAD TO MORE PLOT UPDATE TIME) 
 5. Input Random State for repeatable sampling.
 6. Change tenant by selecting tenant radio-button.
* Click update plot to see changes.
* Open two browser window side by side to compare tenants.
* INFORMATION:
1. Click TWICE on presets to get the slider values.   
2. Note: Gender was predicted using data.
3. After updating the plot, if you see no points, try expanding
   the slider end points.
4. 0 age for a few customers to signify missing DOB (Cant replace
   by mean or other metric since it might have an impact on plot
   more than the current one)

Note 1: Eiffel is missing age values. (Due to GDPR compliance)
Note 2: Data being displayed is static data  from 2019-01-10 to 2020-01-10 (local time)
In case of queries, contact lukish@ridecell.com
""",
width=400, height=80,style=style_dict2)




queries = PreText(text="""""",
width=400, height=100,style=style_dict1)

preset_text = PreText(text="""****PRESETS**** --------------------------------->
Click twice on presets""",
width=400, height=100,style=style_dict4)

page_title=PreText(text="""CARSHARING SEGMENTATION WIDGET (POST COVID OUTBREAK) """,style=style_dict5)

how_to=PreText(text="""""",style=style_dict6)

#widgets_text=PreText(text="""WIDGETS """,style=style_dict6)
widgets_text=PreText(text="""""",style=style_dict6)

#TIll point all good


#p = figure(tooltips=TOOLTIP)

p = figure(tools="pan,wheel_zoom,box_zoom,box_select,reset,save")


#th=['0','1']

th=list(set(df['threshold']))


mapper=factor_cmap('thr', 'Category10_3', th)

def plot_points(p,datapoints_source):

   t = Title()
   t.text = "Car sharing Segmentation Widget"
   p.title=t

   p.scatter("x", "y", source=datapoints_source, legend="thr", alpha=0.5,
          size=12, color=mapper)
   p.xaxis.axis_label = X
   p.yaxis.axis_label = Y
   legend_title = Z+"_Threshold" #Legend Title
   p.legend.title = '> Threshold is 1 else 0 (For Numeric Columns)'


  # p.circle(0, 0, size=0.00000001, color= "#ffffff", legend=legend_title)
   #p.legend.click_policy="hide"



plot_points(p,datapoints_source)

"""
layout = row(
            column(
                widgetbox(slider,width=350),
                widgetbox(sampling,width=350),
                widgetbox(c_sampling,width=350),
                widgetbox(X_sampling,width=350),
                widgetbox(Y_sampling,width=350),
                widgetbox(Xselect,width=350),
                widgetbox(Yselect,width=350),
                widgetbox(Zselect,width=350),
                widgetbox(text_input,width=350),
                widgetbox(bt,width=350),
                widgetbox(radio_button_group,width=350),
                width=400),
            column(p,width=800),
        )

"""



div = Div(text='Tooltips (Double Click on a point to view)',
width=200, height=100)



ldict=dict(c(dfs['threshold']))
ldictkeys=list(ldict.keys())
ldictvalues=list(ldict.values())
ldictlist=['<b>'+str(ldictkeys[f])+'</b>'+': '+str(ldictvalues[f]) for f in range(len(ldictvalues))]
ldictfinal='\n'.join(ldictlist)

plot_text=Div(text='<b style="color:Gray;"></b><br><b style="color:MediumSeaGreen;">Points displayed : Total points in the dataset: </b>'+str(len(dfs))+':'+str(len(df))+'\n'+'<b style="color:MediumSeaGreen;"><br> Counts relative to Threshold: </b>'+ldictfinal,
width=500, height=120)


style_dict8={'font-family': "Times New Roman",'font-size': '15px',
  'font-style': 'bold',
  'color': 'red',
  'width': '600px',
  'height': '20px',
}

style_dict9={'font-family': "Times New Roman",'font-size': '15px',
  'font-style': 'bold',
  'color': 'blue',
  'width': '600px',
  'height': '20px',
}

plot_text2 = PreText(text="""CAUTION! HOVER OVER POINTS OF INTEREST (ONLY) TO REDUCE LOAD!""",
width=400, height=20,style=style_dict8)
  

double_click= PreText(text="""HOVER OVER POINTS OF INTEREST AFTER DOUBLE TAPPING ON PLOT!""",
width=400, height=20,style=style_dict9)


hovertool_widget = RadioButtonGroup(
        labels=["No Hover tool", "Hover tool"], active=0)

hovertool_timer=TextAreaInput(value="", rows=1, title="Select display time for tooltip")

from bokeh.events import DoubleTap
#add a dot where the click happened
import time
def hovercallback(attr, old, new):
 if vdict['hovertool_widget']!=  hovertool_widget.active:  
    if hovertool_widget.active==1:
        p.add_tools(TOOLTIP)
    elif hovertool_widget.active==0:
        del p.tools[-1]
#        p.add_tools(TOOLTIP)
    #print(p.tools)
    
#        time.sleep(float(hovertool_timer.value))
#        del p.tools[-1]
        
    #print(p.tools)
    

 vdict['hovertool_widget']= hovertool_widget.active
    

hovertool_widget.on_change('active',hovercallback)



style_dict1={'font-family': "Times New Roman",
  'font-style': 'bold',
 'color': 'crimson',
  'border': '2px LightGray',
  'border-radius': '25px',
  'border-style': 'ridge',
#  'background': '#73AD21',
  'padding': '5px',
  'width': '350px',
  'height': '75px',
}

hover_warning = PreText(text="""USING HOVERTOOL CAN HANG YOUR BROWSER
IN CASE OF SAMPLES BEING DENSELY POPULATED
(MORE AND CLOSER)""",
width=400, height=100,style=style_dict1)





filtered_df=dfs.copy()
#taptool = p.select(type=TapTool)

#taptool=p.select(type= PolySelectTool)

#taptool=p.select(type=LassoSelectTool)

taptool=p.select(type=BoxSelectTool)



data_table_Columns = [TableColumn(field=Ci, title=Ci) for Ci in filtered_df.columns] # bokeh columns
data_table_source=ColumnDataSource(filtered_df)
data_table = DataTable(columns=data_table_Columns, source=data_table_source,width=2000)

download_button = Button(label="Download", button_type="success")
download_button.js_on_click(CustomJS(args=dict(source=data_table_source),
                            code=open(join(dirname(__file__), "download.js")).read()))

data_table=Panel(child=column(row(data_table,width=1000),row(download_button)),title='Data Table')





glyph_variable=0
def callback(event):
    selected = datapoints_source.selected.indices
    print(selected)
    #global filtered_df
    print(len(filtered_df))
    filtered_df2=filtered_df.iloc[selected,:]
    dictionary=dict(x=filtered_df2[X],
    y=filtered_df2[Y],
    thr=filtered_df2['threshold'])
    for col_name in display_columns:
    # if col_name not in [X,Y]:
      dictionary[col_name]=filtered_df2[col_name]


    
    
    dictionary3={}
    for col_name in filtered_df2.columns:
      dictionary3[col_name]=filtered_df2[col_name]
    data_table_source.data=dictionary3

    ldict=dict(c(filtered_df2['threshold']))
    ldictkeys=list(ldict.keys())
    ldictvalues=list(ldict.values())
    ldictlist=['<b>'+str(ldictkeys[f])+'</b>'+': '+str(ldictvalues[f]) for f in range(len(ldictvalues))]
    ldictfinal='\n'.join(ldictlist)

    plot_text.text='<b style="color:Gray;"></b><br><b style="color:MediumSeaGreen;">Points displayed : Total points in the dataset: </b>'+str(len(filtered_df2))+':'+str(len(df))+'\n'+'<b style="color:MediumSeaGreen;"><br>Counts relative to Threshold: </b>'+ldictfinal

    #pre.text='<h4 style="border-top: 2px solid #778899;width: 1600px"><br><b style="color:slategray">Count of trips: </b>'+str(len(dff))+'<br>'+'<b style="color:slategray">Count of hexes: </b>'+str(len(selected))+'</h4>'
#p.on_event(Tap, callback)

#p.on_event(SelectionGeometry,callback)

p.on_event(PanEnd,callback)





wlayout=column(
                widgetbox(Xselect,width=350,height=70),
                widgetbox(X_sampling,width=350),
                widgetbox(Yselect,width=350,height=70),
                widgetbox(Y_sampling,width=350),
                widgetbox(Zselect,width=350),
                widgetbox(c_sampling,width=350),
                widgetbox(slider,width=350),
                widgetbox(sampling,width=350),
                widgetbox(bt,width=350),
                widgetbox(radio_button_group,width=350),
                widgetbox(hovertool_widget),
                hover_warning,
#                widgetbox(hovertool_timer),
                #widgetbox(div),
                width=400)




tab1 = Panel(child=wlayout, title="Widgets")

dlayout=column(pre,width=600)

tab2 = Panel(child=dlayout, title="Help")

slayout=column(widgetbox(text_input,preset_sampling,width=350))

tab3 = Panel(child=slayout, title="Random Seed")

tabs = Tabs(tabs=[ tab1, tab2, tab3,data_table ])


dpreset = Panel(child=layout(column(dpreset1,dpreset2,dpreset3,dpreset4,dpreset5,dpreset6)), title="Darwin Presets")
epreset = Panel(child=layout(column(epreset1,epreset2,epreset3,epreset4,epreset5,epreset6)), title="Eiffel Presets")

preset_tabs = Tabs(tabs=[ dpreset, epreset ])

layout = column(row(page_title),row(column(how_to,width=1235,height=20),column(widgets_text)),row(column(row(
        column(preset_tabs,width=400),
                    column(row(plot_text,height=70,width=630),row(p),width=700),
            column(tabs),


        )#row(column(preset_text,width=600),column(row(preset1,width=200,height=40),row(preset2,width=200,height=40)))
        ,row()) ) )




"""

layout = column(row(page_title),row(how_to),row(column(row(column(pre,width=600),
                    column(row(plot_text2),row(double_click),row(plot_text,height=70,width=630),row(p),width=630),
            column(
                widgetbox(Xselect,width=350),
                widgetbox(X_sampling,width=350),
                widgetbox(Yselect,width=350),
                widgetbox(Y_sampling,width=350),
                widgetbox(Zselect,width=350),
                widgetbox(c_sampling,width=350),
                widgetbox(slider,width=350),
                widgetbox(sampling,width=350),
                widgetbox(text_input,width=350),
                widgetbox(bt,width=350),
                widgetbox(radio_button_group,width=350),
                #widgetbox(div),
                width=400),


        ),row(column(preset_text,width=600),column(row(preset1,width=200,height=40),row(preset2,width=200,height=40))),row(queries)) ) )

"""

curdoc().add_root(layout)



"""
apps = {'/': Application(FunctionHandler(make_document))}

server = Server(apps, port=5000,host='0.0.0.0')
server.start()
"""
