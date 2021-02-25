import re
import pandas as pd
import string
from pycontractions import Contractions
import gensim.downloader as api
from TweetsGrammarChecker import correct_grammar

model = api.load("glove-twitter-100")
cont = Contractions(kv_model=model)
cont.load_models()

def expand_contractions(string):
    """expand shortened words, e.g. don't to do not"""
    text = list(cont.expand_texts([string], precise=True))[0]
    return text


def remove_emoji(string):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)

# tweets preprocessing function
def clean_dataset(dataset):

    for row_index in dataset.index:
        row_field = dataset.loc[row_index, "Tweet Text"]

        # removing urls
        row_field = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                        '(?:%[0-9a-fA-F][0-9a-fA-F]))+','',  row_field)
        # removing mentions 
        row_field = re.sub("(@[A-Za-z0-9_]+)","", row_field)

        # downcasing text
        row_field = row_field.lower() 

        # partial grammar correction 
        row_field = correct_grammar(row_field)

        # expand contractions 
        row_field = expand_contractions(row_field)
        row_field = row_field.replace("I", "i") 

        # removing possessive pronouns 
        row_field = re.sub("'s", "", row_field)
        row_field = re.sub("’s", "", row_field)

        # removing punctuation signs 
        for punct_sign in string.punctuation:
            row_field = row_field.replace(punct_sign, " ")
            row_field = row_field.replace("’", " ") 
            row_field = row_field.replace("”", " ")
            row_field = row_field.replace("“", " ")
            row_field = row_field.replace("\n", " ")

        # removing numbers  
        row_field = re.sub(r'\d+', '', row_field)

        # removing emojis
        row_field = remove_emoji(row_field)
    
        # downcasing text
        row_field = row_field.lower() 

        if(len(row_field) > 0):
            dataset.loc[row_index, "Tweet Text"] = row_field
        else: 
            dataset.drop(index = row_index, inplace=True)
    
    # removing useless info from data
    #dataset.drop(columns=["Unnamed: 0", "Unnamed: 0.1", "Tweet Language"], inplace=True)

    #print(dataset.head(5))

    return dataset 

# loading input data
dataset = pd.read_csv("./Dataset/samsung_week_5_to_12_October_tagged.csv")
cleaned_dataset = clean_dataset(dataset)
cleaned_dataset.to_csv("./Dataset/samsung_week_5_to_12_October_tagged_pp.csv")