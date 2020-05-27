#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 11:47:09 2020

@author: lukishyadav
"""

import pandas as pd
import numpy as np
import holoviews as hv
from holoviews import opts, dim
hv.extension('bokeh')


macro_df = pd.read_csv('http://assets.holoviews.org/macro.csv', '\t')
key_dimensions   = [('year', 'Year'), ('country', 'Country')]
value_dimensions = [('unem', 'Unemployment'), ('capmob', 'Capital Mobility'),
                    ('gdp', 'GDP Growth'), ('trade', 'Trade')]
macro = hv.Table(macro_df, key_dimensions, value_dimensions)



gdp_curves = macro.to.curve('Year', 'GDP Growth')
gdp_unem_scatter = macro.to.scatter('Year', ['GDP Growth', 'Unemployment'])
annotations = hv.Arrow(1973, 8, 'Oil Crisis', 'v') * hv.Arrow(1975, 6, 'Stagflation', 'v') *\
hv.Arrow(1979, 8, 'Energy Crisis', 'v') * hv.Arrow(1981.9, 5, 'Early Eighties\n Recession', 'v')


composition=(gdp_curves * gdp_unem_scatter* annotations)
composition.opts(
    opts.Curve(color='k'), 
    opts.Scatter(cmap='Blues', color='Unemployment', 
                 line_color='k', size=dim('Unemployment')*1.5),
    opts.Text(text_font_size='13px'),
    opts.Overlay(height=400, show_frame=False, width=700))


hv.save(composition, 'holomap.html')