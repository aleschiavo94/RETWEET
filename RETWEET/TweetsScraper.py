import tweepy
import pandas as pd
import time
import csv
from time import sleep 

# Function that gets US states coordinates from csv file and saves them in dict
def build_coords_dictionary():
    with open('./Utils/US_States_Coordinates.csv') as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=",")
        coordinates = {}
        for row in csv_reader:
            str = row["latitude"] + "," + row["longitude"] + "," + "350km"
            coordinates[row["name"]] = str
        
        return coordinates 

# Function created to extract coordinates from tweet if it has coordinate info
def extract_coordinates(row):
    if row['Tweet Coordinates']:
        return row['Tweet Coordinates']['coordinates']
    else:
        return None

# Function created to extract place such as city, state or country from tweet if it has place info
def extract_place(row):
    if row['Place Info']:
        return row['Place Info'].full_name
    else:
        return None
        
# Tweets scraping function 
def srapeTweets(keywords, coordinates, numTweets):
    
    program_start = time.time()
    tweets_list = []
    # Creation of dataframe 
    tweets_df = pd.DataFrame(columns=['Tweet Id', 
                                      'Tweet Datetime',
                                      'Tweet Text',
                                      'Twitter @ Name', 
                                      'Tweet Coordinates',
                                      'Place Info', 
                                      'User Location', 
                                      'Tweet Language'
                                    ])

    for state, coords in coordinates.items():
        start_run = time.time()

        # collect Tweets
        print("Looking for Tweets around [{}] : [{}] ...\n".format(state, coords))
        tweets = tweepy.Cursor(api.search, 
                            q=keywords,
                            result_type="recent",
                            geocode=coords,
                            lang="en",
                            tweet_mode="extended"
                            ).items(numTweets)

        f = open("./Utils/usernames_blacklist.txt", "r", encoding="utf8")
        f2 = open("./Utils/texts_blacklist.txt", "r", encoding="utf8")
        
        # Pulling information from each Tweet
        usernames = f.read()
        texts = f2.read()

        for tweet in tweets:
            if(tweet.user.screen_name in usernames or texts in tweet.full_text):
                continue

            tweets_list.append([tweet.id_str, 
                        tweet.created_at,
                        tweet.full_text, 
                        tweet.user.screen_name,
                        tweet.coordinates, 
                        tweet.place, 
                        tweet.user.location,
                        tweet.lang
                        ])
            
        # run ended
        end_run = time.time()
        duration_run = round((end_run-start_run)/60, 2)
        
        tweetsScraped = len(tweets_list)
        print('no. of tweets scraped is {}'.format(tweetsScraped))
        print('time taken to complete is {} mins'.format(duration_run))
    
    # Creation of dataframe from tweets_list
    tweets_df = pd.DataFrame(tweets_list,columns=['Tweet Id', 
                                                    'Tweet Datetime',
                                                    'Tweet Text',
                                                    'Twitter @ Name', 
                                                    'Tweet Coordinates',
                                                    'Place Info', 
                                                    'User Location', 
                                                    'Tweet Language'
                                                    ])
         
    # sorting tweets by id 
    tweets_df.sort_values("Tweet Id", inplace = True) 
  
    # dropping duplicates by Tweet Id
    tweets_df.drop_duplicates(subset ="Tweet Id", 
                    keep = "first", inplace = True) 
    
    # dropping duplicates by Tweet Text
    tweets_df.drop_duplicates(subset ="Tweet Text", 
                    keep = "first", inplace = True) 

    # Checks if there are coordinates attached to tweets, if so extracts them
    tweets_df['Tweet Coordinates'] = tweets_df.apply(extract_coordinates,axis=1)

    # Checks if there is place information available, if so extracts them
    tweets_df['Place Info'] = tweets_df.apply(extract_place,axis=1)

    # store scraped tweets in csv file
    tweets_df.to_csv('samsung_week_5_to_12_October.csv')

    program_end = time.time()
    totScraped = len(tweets_df.index)
    print("---------------------------------------------")
    total_duration = round((program_end - program_start)/60, 2)
    print('Total no. of tweets without duplicates is [{}]'.format(totScraped))
    print('time taken to complete is [{}] mins'.format(total_duration))


def twitter_init():
    # Create variables for each key, secret, token
    consumer_key = "IG3sIHi0ovFvy7ZVygb928qzu"
    consumer_secret = "UXCsm2QYnLTWvliXTq9CPUPgY6oiDFPrWeOFY1u3x6Wv3SX0A5"
    access_token = "1301906875454304263-CQ8EKyDnuxymqez9EWtznKwQXhDZUH"
    access_token_secret = "Yln6i4Mv6PXAmZY8Vz22RGW86PeW6b1y4YZsIEyGDHwcM"

    # Set up OAuth and integrate with API
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True) 

    return api

# -----------------------------------------------------------------------------

api = twitter_init()

# setting query variables 
keywords_file = open("./Utils/samsung_keywords.txt", "r")
keywords = keywords_file.read()
coordinates = build_coords_dictionary()
numTweets = 500

# tweets scraping
srapeTweets(keywords, coordinates, numTweets)

