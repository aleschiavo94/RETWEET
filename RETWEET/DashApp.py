import pandas as pd
import mysql.connector
import time
import datetime
import joblib

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

import plotly.express as px


import plotly.offline as py
from plotly.subplots import make_subplots

import re
import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from TweetsTextRepresentation import textElaboration

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import itertools
import math
import base64
from flask import Flask
import os
import psycopg2

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = 'Real-Time Twitter Monitor'
server = app.server

app.layout = html.Div(style={'padding': '20px'}, children=[
    html.H2('Real-time Twitter Sentiment Analysis for Brand Improvement', style={
        'textAlign': 'center'
    }),    

    html.Div(id='live-update-graph'),
    html.Div(id='live-update-graph-bottom'),
    
    # ABOUT ROW
    html.Div(
        className='row',
        children=[
            html.Div(
                className='three columns',
                children=[
                    html.P(
                    'Data extracted from:'
                    ),
                    html.A(
                        'Twitter API',
                        href='https://developer.twitter.com'
                    )                    
                ]
            ),
            html.Div(
                className='three columns',
                children=[
                    html.P(
                    'Code avaliable at:'
                    ),
                    html.A(
                        'GitHub',
                        href=''
                    )                    
                ]
            ),
            html.Div(
                className='three columns',
                children=[
                    html.P(
                    'Made with:'
                    ),
                    html.A(
                        'Dash / Plot.ly',
                        href='https://plot.ly/dash/'
                    )                    
                ]
            ),
            html.Div(
                className='three columns',
                children=[
                    html.P(
                    'Author:'
                    ),
                    html.A(
                        'Alessio Schiavo',
                        href=''
                    )                    
                ]
            )                                                          
        ], style={'marginLeft': 70, 'fontSize': 16}
    ),

    dcc.Interval(
        id='interval-component-slow',
        interval=1200*1000, # 20 minutes in milliseconds
        n_intervals=1
    )
    ])

# init variables
values = {"negatives": 0,
          "neutrals": 0,
          "positives": 0}
negatives = 0
neutrals = 0
positives = 0
vocabulary = []
lemmatized_dataset = ""

# Multiple components can update everytime interval gets fired.
@app.callback(Output('live-update-graph', 'children'),
              [Input('interval-component-slow', 'n_intervals')])
def update_graph_live(n):
    global values
    global negatives
    global neutrals
    global positives
    global vocabulary
    global lemmatized_dataset

    # connect to MySQL DB 
    db_connection = mysql.connector.connect(
        host="localhost",
        user="root",
        passwd="root",
        database="Twitterdb",
        charset = 'utf8'
    )

    time_now = datetime.datetime.utcnow()
    time_10mins_before = datetime.timedelta(days=0,hours=0,minutes=20)
    time_interval = time_now - time_10mins_before
    time_interval_f = time_interval.strftime('%Y-%m-%d %H:%M:%S')
    print(time_interval_f)

    # ------------------------- retrieve data from MySQL DB ------------------------- #
    query = "SELECT tweet_id, tweet_datetime, tweet_text, tweet_username FROM {}  \
            WHERE tweet_datetime >= '{}'" .format("samsungtweets", time_interval_f)
    df = pd.read_sql(query, con=db_connection)


    # ------------------------- tokenization + stopwords filtering + lemmatization ------------------------- #
    lemmatized_dataset = textElaboration(df, "tweet_text")

    # ------------------------- load best classifier from disk and predict tweets polarities ------------------------- #
    loaded_model = joblib.load('./Tuned_Models/RF_0.pkl')
    polarities = loaded_model.predict(df["tweet_text"])
    #print(polarities)

    # label coding 
    tag_codes = {
        "positive" : 1, 
        "negative" : 0,
        "neutral" : -1
    }

    df["polarity"] = polarities

    # ------------------------- Time series tweets graph ------------------------- #
    result = df.groupby([pd.Grouper(key='tweet_datetime', freq='2min'), 'polarity']).count().unstack(fill_value=0).stack().reset_index()
    result = result.rename(columns={"tweet_id": "Num of '{}' mentions".format("samsung"), "tweet_datetime":"Time in UTC"})  
    time_series = result["Time in UTC"][result['polarity']==0].reset_index(drop=True)

    # ------------------------- donut-like pie chart ------------------------- #
    for value in polarities:
        # neutral tweet
        if value == -1:
            neutrals += 1
        
        # negative tweet
        if value == 0:
            negatives += 1

        # positive tweet 
        if value == 1:
            positives += 1

    values["neutrals"] += neutrals
    values["negatives"] += negatives
    values["positives"] += positives

    labels = ['Neutral','Negative','Positive']
    colors = ['039be5',"e53935","#00897b"]

    # ------------------------- Words Frequencies Bar Chart ------------------------- #
    for tweet in lemmatized_dataset:
        tweet_words = tweet.split(" ")
        #print(tweet_words)
        for word in tweet_words:
            if len(word) > 1:
                vocabulary.append(word.lower())

    # get top 10 words 
    fdist = FreqDist(vocabulary)
    fd = pd.DataFrame(fdist.most_common(11), columns = ["Word","Frequency"]).drop([0]).reindex()

    # Create the graph 
    children = [
            html.Div([
                html.Div([
                    dcc.Graph(
                        id='crossfilter-indicator-scatter',
                        figure={
                            'data': [
                                go.Scatter(
                                    x=time_series,
                                    y=result["Num of '{}' mentions".format("samsung")][result['polarity']==-1].reset_index(drop=True),
                                    name="Neutrals",
                                    opacity=0.8,
                                    mode='lines',
                                    line=dict(width=0.5, color='rgb(131, 90, 241)'),
                                    stackgroup='one' 
                                ),
                                go.Scatter(
                                    x=time_series,
                                    y=result["Num of '{}' mentions".format("samsung")][result['polarity']==0].reset_index(drop=True).apply(lambda x: -x),
                                    name="Negatives",
                                    opacity=0.8,
                                    mode='lines',
                                    line=dict(width=0.5, color='rgb(255, 50, 50)'),
                                    stackgroup='two' 
                                ),
                                go.Scatter(
                                    x=time_series,
                                    y=result["Num of '{}' mentions".format("samsung")][result['polarity']==1].reset_index(drop=True),
                                    name="Positives",
                                    opacity=0.8,
                                    mode='lines',
                                    line=dict(width=0.5, color='rgb(184, 247, 212)'),
                                    stackgroup='three' 
                                )
                            ]
                        } 
                    )
                ], style={'width': '70%', 'display': 'inline-block', 'padding': '0 0 0 20'}),

            html.Div([
                dcc.Graph(
                    id='pie-chart',
                    figure={
                        'data': [
                            go.Pie(
                                labels=labels, 
                                values=[values["positives"], values["negatives"], values["neutrals"]],
                                name="View Metrics",
                                marker_colors=['rgba(184, 247, 212, 0.6)','rgba(255, 50, 50, 0.6)','rgba(131, 90, 241, 0.6)'],
                                textinfo='value',
                                hole=.65)
                        ],
                        'layout':{
                            'showlegend':False,
                            'title':'Total Tweets per category',
                            'annotations':[
                                dict(
                                    text='{0:.1f}K'.format((values["positives"] + values["negatives"] + values["neutrals"])/1000),
                                    font=dict(
                                        size=40
                                    ),
                                    showarrow=False
                                )
                            ]
                        }

                    }
                )
            ], style={'width': '27%', 'display': 'inline-block'}),

            html.Div(
                    className='row',
                    children=[
                        html.Div(
                            children=[
                                html.P('% Tot Positive Tweets',
                                    style={
                                        'fontSize': 17
                                    }
                                ),
                                html.P("{}%".format(round((values["positives"]/(values["positives"] + values["negatives"] + values["neutrals"])*100),2)),
                                    style={
                                        'fontSize': 40
                                    }
                                )
                            ], 
                            style={
                                'width': '15%', 
                                'display': 'inline-block'
                            }

                        ),
                        html.Div(
                            children=[
                                html.P('% Tot Negative Tweets',
                                    style={
                                        'fontSize': 17
                                    }
                                ),
                                html.P("{}%".format(round((values["negatives"]/(values["positives"] + values["negatives"] + values["neutrals"])*100),2)),
                                    style={
                                        'fontSize': 40
                                    }
                                )
                            ], 
                            style={
                                'width': '15%', 
                                'display': 'inline-block'
                            }
                        ),
                        html.Div(
                            children=[
                                html.P('% Tot Neutral Tweets',
                                    style={
                                        'fontSize': 17
                                    }
                                ),
                                html.P("{}%".format(round((values["neutrals"]/(values["positives"] + values["negatives"] + values["neutrals"])*100),2)),
                                    style={
                                        'fontSize': 40
                                    }
                                )
                            ], 
                            style={
                                'width': '15%', 
                                'display': 'inline-block'
                            }
                        ),

                        html.Div(
                            children=[
                                html.P('Tot # of Tweets Today',
                                    style={
                                        'fontSize': 17
                                    }
                                ),
                                html.P("{}".format((values["positives"] + values["negatives"] + values["neutrals"])),
                                    style={
                                        'fontSize': 40
                                    }
                                )
                            ], 
                            style={
                                'width': '15%', 
                                'display': 'inline-block'
                            }
                        ),

                        html.Div(
                            children=[
                                html.P("Currently tracking \"Samsung\" brand on Twitter.",
                                    style={
                                        'fontSize': 25
                                    }
                                ),
                            ], 
                            style={
                                'width': '35%', 
                                'display': 'inline-block'
                            }
                        ),

                    ],
                    style={'marginLeft': 130}
                ),

            html.Div([
                dcc.Graph(
                    id='x-time-series',
                    figure = {
                        'data':[
                            go.Bar(                          
                                x=fd["Frequency"].loc[::-1],
                                y=fd["Word"].loc[::-1], 
                                name="Neutrals", 
                                orientation='h',
                                marker_color='rgba(131, 90, 241, 0.6)',
                                marker=dict(
                                    line=dict(
                                        color='rgba(131, 90, 241, 0.6)',
                                        width=1),
                                    ),
                            )
                        ],
                        'layout':{
                            'hovermode':"closest"
                        }
                    }        
                )
            ], style={'width': '70%', 'display': 'inline-block', 'padding': '20 0 0 20'}),


        ]),
    ]

    return children



if __name__ == '__main__':
    app.run_server(debug=True)