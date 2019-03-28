import plotly.plotly as py
import plotly.graph_objs as go
import plotly
import pandas as pd
import plotly.io as pio
print(plotly.__version__)
from IPython.display import Image
# Read data from a csv
z_data = pd.read_csv('output/training_landscape.txt', sep = ' ', header = None).as_matrix()

trace =  [
    go.Heatmap(
        z = z_data[:,2],
	x = z_data[:,0],
	y = z_data[:,1],
	zsmooth = 'best'
    )
]
data = trace
layout =  go.Layout(
    xaxis=dict(
        title=r'$\Large \alpha$',
type='log',
    ),
    yaxis=dict(
        title= r'$\Large \theta$',
type='log',
    )
)

#plotly.offline.plot({"data": data, "layout": layout}, auto_open = True)

fig = go.Figure(data = data, layout = layout)
pio.write_image(fig, 'fitting_landscape.pdf')

#############################

z_data = pd.read_csv('output/testing_landscape.txt', sep = ' ', header = None).as_matrix()

trace =  [
    go.Heatmap(
        z = z_data[:,2],
	x = z_data[:,0],
	y = z_data[:,1],
	zsmooth = 'best'
    )
]
data = trace
layout =  go.Layout(
    xaxis=dict(
        title=r'$\Large \alpha$',
type='log',
    ),
    yaxis=dict(
        title= r'$\Large \theta$',
type='log',
    )
)

#plotly.offline.plot({"data": data, "layout": layout}, auto_open = True)

fig = go.Figure(data = data, layout = layout)
pio.write_image(fig, 'forecast_landscape.pdf')

