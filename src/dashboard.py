# Import necessary libraries
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
from flask import Flask
from dash.dependencies import Input, Output

# Load data
fraud_data = pd.read_csv('../data/Fraud_Data.csv')

# Initialize the Flask server and then Dash
server = Flask(__name__)
app = Dash(__name__, server=server, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Data preprocessing for dashboard
fraud_data['purchase_time'] = pd.to_datetime(fraud_data['purchase_time'])
fraud_data['signup_time'] = pd.to_datetime(fraud_data['signup_time'])
fraud_data['day'] = fraud_data['purchase_time'].dt.date
fraud_data['fraud'] = fraud_data['class'].apply(lambda x: 'Fraud' if x == 1 else 'Not Fraud')

# Layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Fraud Detection Dashboard", className="text-center"), width=12)
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id='fraud-timeseries'), width=6),
        dbc.Col(dcc.Graph(id='fraud-geography'), width=6)
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id='device-browser-fraud'), width=12)
    ])
])

# Callbacks
@app.callback(
    Output('fraud-timeseries', 'figure'),
    Output('fraud-geography', 'figure'),
    Output('device-browser-fraud', 'figure'),
    Input('fraud-timeseries', 'id')
)
def update_charts(_):
    # Time series fraud cases
    timeseries_fig = px.line(fraud_data.groupby('day').size().reset_index(name='Counts'),
                             x='day', y='Counts', title='Fraud Cases Over Time')

    # Fraud by country
    geo_fig = px.choropleth(fraud_data, locations="country", locationmode="country names",
                            color="fraud", hover_name="country", title='Fraud by Geography')

    # Fraud by Device and Browser
    device_browser_fig = px.histogram(fraud_data, x='browser', color='fraud',
                                      title="Fraud Cases by Device and Browser", barmode='group')
    return timeseries_fig, geo_fig, device_browser_fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
