#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  9 22:06:13 2020

@author: lukishyadav
"""

from bokeh.io import curdoc
from os.path import dirname, join
from bokeh.models.widgets import FileInput
from pybase64 import b64decode
import pandas as pd
import io
from bokeh.models.widgets import Button, RadioButtonGroup, Select, Slider,TextInput,TextAreaInput
from bokeh.layouts import column,layout,row,widgetbox

from bokeh.plotting import figure, output_file, show

from datetime import date
from random import randint

from bokeh.io import output_file, show
from bokeh.models import ColumnDataSource, DataTable, DateFormatter, TableColumn

from bokeh.models import (Button, ColumnDataSource, CustomJS, DataTable,
                          NumberFormatter, RangeSlider, TableColumn,)

df=pd.read_csv('darwin_rental_data.csv')

df=df.head(10000)

data = dict(
        dates=[date(2014, 3, i+1) for i in range(10)],
        downloads=[randint(0, 100) for i in range(10)],
    )
source = ColumnDataSource(df)
"""
columns = [
        TableColumn(field="dates", title="Date", formatter=DateFormatter()),
        TableColumn(field="downloads", title="Downloads"),
    ]
data_table = DataTable(source=source, columns=df.columns, width=400, height=280)

"""

#from bokeh.models.widgets import DataTable, DateFormatter, TableColumn

Columns = [TableColumn(field=Ci, title=Ci) for Ci in df.columns] # bokeh columns
data_table = DataTable(columns=Columns, source=ColumnDataSource(df),width=800) # bokeh table

#show(data_table)

#show(data_table)

    
button = Button(label="Download", button_type="success")
button.js_on_click(CustomJS(args=dict(source=source),
                            code=open(join(dirname(__file__), "download.js")).read()))
#bt.on_click(update_click) 



    

doc=curdoc()
doc.add_root(layout(row(data_table),button))