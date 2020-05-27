#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 10:32:04 2020

@author: lukishyadav
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 10:19:21 2020

@author: lukishyadav
"""

import numpy as np
import holoviews as hv
from holoviews.operation import histogram
hv.extension('bokeh')


renderer = hv.renderer('bokeh')

data = np.random.multivariate_normal((0, 0), [[1, 0.1], [0.1, 1]], (1000,))
points = hv.Points(data)

points.hist(dimension=['x','y'])

layout = points

doc = renderer.server_doc(layout)
doc.title = 'HoloViews App'



