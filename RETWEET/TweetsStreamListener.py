import tweepy 
import pandas as pd 
import mysql.connector
import csv

from pycontractions import Contractions
import gensim.downloader as api
from TweetsPreProcessing import clean_dataset

model = api.load("glove-twitter-100")
cont = Contractions(kv_model=model)
cont.load_models()

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

keywords_file_1 = open("./Utils/samsung_keywords_realtime_1.txt", "r")
keywords_file_2 = open("./Utils/samsung_keywords_realtime_2.txt", "r")
keywords_file_3 = open("./Utils/samsung_keywords_realtime_3.txt", "r")
keywords_1 = keywords_file_1.read().split(",")
keywords_2 = keywords_file_2.read().split(",")
keywords_3 = keywords_file_3.read().split(",")

def keywords_filtering(text, keywords_1, keywords_2, keywords_3):
    # keywords filtering as in offline scraping 
    for keyword_1 in keywords_1:
        if keyword_1 in text:
            for keyword_2 in keywords_2:
                if keyword_2 in text:
                    for keyword_3 in keywords_3:
                        if keyword_3 in text:
                            return False
                    return True

f = open("./Utils/usernames_blacklist.txt", "r", encoding="utf8")
f2 = open("./Utils/texts_blacklist.txt", "r", encoding="utf8")
black_users = f.read()
black_texts = f2.read()

def blacklists_filtering(black_users, black_texts, tweet_username, tweet_text):
    if(tweet_username in black_users or black_texts in tweet_text):
        print("Found Blacklisted...")
        return True
    else:
        return False

# functions to get list of tweets places in US 
def build_locations():
    with open('./Utils/US_States_BBoxes.csv') as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=",")
        locations = []
        counter = 0
        for row in csv_reader:
            counter += 1
            if counter < 25:
                locations.append(float(row["xmin"]))
                locations.append(float(row["ymin"]))
                locations.append(float(row["xmax"]))
                locations.append(float(row["ymax"]))
        
        return locations

# DB table settings 
TRACK_WORDS = ['Samsung']
TABLE_NAME = "SamsungTweets"
TABLE_NAME_2 = "samsungtweets_tagged"
TABLE_ATTRIBUTES = "tweet_id VARCHAR(255), \
                    tweet_datetime DATETIME, \
                    tweet_text VARCHAR(500), \
                    tweet_username VARCHAR(255)"
                    #tweet_location VARCHAR(255)"
TABLE_ATTRIBUTES_2 = "tweet_id VARCHAR(255), \
                      tweet_datetime DATETIME, \
                      tweet_text VARCHAR(500), \
                      tweet_username VARCHAR(255), \
                      tweet_tag INT"                   

print("Connecting to MySQL db...")
# connect to MySQL DB 
mydb = mysql.connector.connect(
    auth_plugin='mysql_native_password',
    host="localhost",
    user="root",
    passwd="root",
    database="twitterdb",
    charset = 'utf8'
)
print("Successfully Connected!")

# create table if it doesn't exist 
if mydb.is_connected():
    mycursor = mydb.cursor()
    mycursor.execute("""
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_name = '{0}'
        """.format(TABLE_NAME))
    if mycursor.fetchone()[0] != 1:
        mycursor.execute("CREATE TABLE {} ({})" \
            .format(TABLE_NAME, TABLE_ATTRIBUTES))
        mydb.commit()
    mycursor.close()

# create table if it doesn't exist 
if mydb.is_connected():
    mycursor = mydb.cursor()
    mycursor.execute("""
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_name = '{0}'
        """.format(TABLE_NAME_2))
    if mycursor.fetchone()[0] != 1:
        mycursor.execute("CREATE TABLE {} ({})" \
            .format(TABLE_NAME_2, TABLE_ATTRIBUTES_2))
        mydb.commit()
    mycursor.close()

api = twitter_init()

tweet_dict = {"tweet_id": [],
              "tweet_datetime": [],
              "Tweet Text": [],
              "tweet_username": [],
              "tweet_location": []}

tweets_counter = 0
batch_counter = 0

# override tweepy.StreamListener to add logic to on_status
class MyStreamListener(tweepy.StreamListener):

    def on_status(self, status, tweet_mode="extended"):

        global tweets_counter
        global tweet_dict
        global batch_counter

        # get tweet info 
        id_str = status.id_str
        user_created_at = status.created_at

        try:
            text = status.extended_tweet["full_text"]
        except:
            text = status.text

        coords = status.coordinates 
        user = status.user 
        user_location = user.location
        username = user.screen_name
        lang = status.lang

        # discard if retweeted tweet
        if "RT" in text or lang != "en":
            return True

        # filter blacklist users and texts
        if blacklists_filtering(black_users, black_texts, username, text):
            return True

        # tweets keywords filtering
        if keywords_filtering(text, keywords_1, keywords_2, keywords_3):
                        
            tweets_counter += 1
            print("{} tweets found...".format(tweets_counter))

            # print tweet info 
            #print(text)
            #print("Username: {}".format(username))
            #print("Created at: {}".format(user_created_at))
            #print("Coords: {}" .format(coords))
            #print("Location: {}" .format(user_location))
            
            # add tweet to pandas dataframe
            tweet_dict["tweet_id"].append(id_str)
            tweet_dict["tweet_datetime"].append(user_created_at)
            tweet_dict["Tweet Text"].append(text)
            tweet_dict["tweet_username"].append(username)
            tweet_dict["tweet_location"].append(user_location)

            # preprocess and store in DB n-tweets batch
            if tweets_counter == 10:
                batch_counter += 1
                print("Preprocessing and storing into DB tweet batch #{}..." .format(batch_counter))

                #save tweets into pandas DataFrame 
                tweet_df = pd.DataFrame(data=tweet_dict)
                
                # tweets preprocessing
                tweet_cleaned = clean_dataset(tweet_df)
                #print(tweet_cleaned)

                # store tweets into MySQL DB 
                if mydb.is_connected():
                    
                    datetime_index = 0
                    for row_index in tweet_cleaned.index:
                        mycursor = mydb.cursor()

                        id_str = tweet_cleaned.loc[row_index, "tweet_id"]
                        user_created_at = tweet_dict["tweet_datetime"][datetime_index]
                        cleaned_text = tweet_cleaned.loc[row_index, "Tweet Text"]
                        username = tweet_cleaned.loc[row_index, "tweet_username"]
                        #user_location = tweet_cleaned.loc[row_index, "tweet_location"]

                        sql = "INSERT INTO {} (tweet_id, tweet_datetime, tweet_text, tweet_username) VALUES \
                            (%s, %s, %s, %s)" .format(TABLE_NAME)
                        val = (id_str, user_created_at, cleaned_text, username)     
                        mycursor.execute(sql, val)
                        mydb.commit()
                        mycursor.close()
                    
                        datetime_index += 1

                    # reset tweets_counter and dict 
                    tweet_dict = {"tweet_id": [],
                                    "tweet_datetime": [],
                                    "Tweet Text": [],
                                    "tweet_username": [],
                                    "tweet_location": []}

                    tweets_counter = 0

            return True
    

# creating a stream
myStreamListener = MyStreamListener()
myStream = tweepy.Stream(auth = api.auth, listener=myStreamListener, tweet_mode="extended")

locations = [-87.634938,24.523096,-80.031362,31.000888]

# starting a stream
print("Listening for tweets...")
myStream.filter(languages=["en"], track=["samsung"])

# Close the MySQL connection as it finished
# However, this won't be reached as the stream listener won't stop automatically
# Press STOP button to finish the process.
mydb.close()
   