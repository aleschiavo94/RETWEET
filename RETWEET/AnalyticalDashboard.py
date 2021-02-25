import pandas as pd
import mysql.connector
import time
import datetime
import joblib

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

import plotly.express as px
import datetime

import plotly.offline as py
import plotly.graph_objs as go
from plotly.subplots import make_subplots

import re
import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from TweetsTextRepresentation import textElaboration

# init variables
values = {"negatives": 0,
          "neutrals": 0,
          "positives": 0}
negatives = 0
neutrals = 0
positives = 0
vocabulary = []

while(True):
    # connect to MySQL DB 
    db_connection = mysql.connector.connect(
        host="localhost",
        user="root",
        passwd="root",
        database="Twitterdb",
        charset = 'utf8'
    )

    time_now = datetime.datetime.utcnow()
    time_10mins_before = datetime.timedelta(hours=10,minutes=0)
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

    # ------------------------- store tagged data into MySQL DB ------------------------- #
    if db_connection.is_connected():
        for row_index in df.index:
            mycursor = db_connection.cursor()

            id_str = int(df.loc[row_index, "tweet_id"])
            user_created_at = str(df.loc[row_index, "tweet_datetime"])
            cleaned_text = df.loc[row_index, "tweet_text"]
            username = df.loc[row_index, "tweet_username"]
            polarity = int(polarities[row_index])

            sql = "INSERT INTO {} (tweet_id, tweet_datetime, tweet_text, tweet_username, tweet_tag) VALUES \
                (%s, %s, %s, %s, %s)" .format("samsungtweets_tagged")
            val = (id_str, user_created_at, cleaned_text, username, polarity)     
            mycursor.execute(sql, val)
            db_connection.commit()
            mycursor.close()

    # label coding 
    tag_codes = {
        "positive" : 1, 
        "negative" : 0,
        "neutral" : -1
    }

    df["polarity"] = polarities

    fig = make_subplots(rows=2, cols=2, 
                        subplot_titles=("Totals Table", 
                                        "Total % shares",
                                        "Tweets in time", 
                                        "Top tweeted words"),
                        specs=[[{"type": "table"} , {"type": "domain"}],
                            [{"type": "scatter"}, {"type": "bar"}]]
                    )

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

    fig.add_trace(go.Pie(labels=labels, values=[values["neutrals"], values["negatives"], values["positives"]], hole=0.3),1, 2,)
    fig.update_traces(marker=dict(colors=colors, line=dict(color='#f5f5f5', width=0)))

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
    
    fig.add_trace(go.Bar(x=fd["Word"], y=fd["Frequency"], name="Freq Dist", width=0.6), row=2, col=2)
    fig.update_traces(marker_color='#2979ff', marker_line_color='#2979ff', \
            marker_line_width=0.5, opacity=1, row=2, col=2)

    #fig.update_layout(xaxis_tickangle=-90)

    # ------------------------- Time series tweets graph ------------------------- #
    result = df.groupby([pd.Grouper(key='tweet_datetime', freq='5min'), 'polarity']).count().unstack(fill_value=0).stack().reset_index()
    result = result.rename(columns={"tweet_id": "Num of '{}' mentions".format("samsung"), "tweet_datetime":"Time in UTC"})  
    time_series = result["Time in UTC"][result['polarity']==0].reset_index(drop=True)

    fig.add_trace(go.Scatter(
        x=time_series,
        y=result["Num of '{}' mentions".format("samsung")][result['polarity']==0].reset_index(drop=True),
        name="Negative",
        line_color="#e53935",
        opacity=0.8), row=2, col=1)   

    fig.add_trace(go.Scatter(
        x=time_series,
        y=result["Num of '{}' mentions".format("samsung")][result['polarity']==-1].reset_index(drop=True),
        name="Neutral",
        line_color="#039be5",
        opacity=0.8), row=2, col=1)
        
    fig.add_trace(go.Scatter(
        x=time_series,
        y=result["Num of '{}' mentions".format("samsung")][result['polarity']==1].reset_index(drop=True),
        name="Positive",
        line_color="#00897b",
        opacity=0.8), row=2, col=1)

    # ------------------------- Cumulative Polarities Table ------------------------- #
    fig.add_trace((go.Table(
        columnwidth = [500,500],
        header=dict(values=['Negatives Total', 'Positives Total', 'Neutrals Total'],
                    line_color='darkslategray',
                    fill_color=['#e53935',"#00897b","#039be5"],
                    font=dict(color='black', size=20),
                    align='center',
                    height=60),
        cells=dict(values=[values["negatives"], # negatives
                        values["positives"], # positives
                        values["neutrals"]], # neutrals
                line_color='darkslategray',
                font=dict(color='black', size=32),
                fill_color='#eeeeee',
                height=80,
                align='center'))
    ), 
    row=1, col=1)

    # Rotate x-axis labels
    #fig.update_xaxes(tickangle=45)

    # Set theme, margin
    fig.update_layout(
        title_text= "Real-time tracking '{}' mentions on Twitter {} UTC".format("samsung" ,datetime.datetime.utcnow().strftime('%m-%d %H:%M')),
        template="plotly_dark",
        margin=dict(r=150, t=100, b=50, l=20)

    )

    # display plots
    fig.show()

    # update every 3 minutes
    time.sleep(180)
    print("Updating Dashboard...")