import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly
import pandas as pd
import plotly.io as pio
print(plotly.__version__)
from IPython.display import Image
# Read data from a csv

def landscape(z_data):
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
            #type='log',
        ),
        yaxis=dict(
            title= r'$\Large \theta$',
            #type='log',
        )
    )

    #plotly.offline.plot({"data": data, "layout": layout}, auto_open = True)

    fig = go.Figure(data = data, layout = layout)
    iplot(fig)
    #static_image_bytes = pio.to_image(fig, format='pdf')
    #Image(static_image_bytes)
