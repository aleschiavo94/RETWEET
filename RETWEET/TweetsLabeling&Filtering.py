import re 
import pandas as pd 

# loading input data
dataset = pd.read_csv("./Dataset/samsung_week_5_to_12_October.csv")
print("Dataset loaded")
print("Numero righe: {}" .format(len(dataset.index)))

spam_usernames = []

for row_index in dataset.index:
    row_field = dataset.loc[row_index, "Tweet Text"]

    try:
        print("row #[{}]:".format(row_index))
        print(row_field)
    except:
        continue

    repeat = True 
    while(repeat):
        label = input()

        # positive tweet
        if(label == 'p'):
            dataset.loc[row_index, "tag"] = "positive"
            repeat = False
        
        # negative tweet 
        if(label == 'n'):
            dataset.loc[row_index, "tag"] = "negative"
            repeat = False

        # neutral tweet 
        if(label == 'u'):
            dataset.loc[row_index, "tag"] = "neutral"
            repeat = False

        # tweet to be deleted 
        if(label == 'd'):
            dataset.drop(index=row_index, inplace=True)
            repeat = False
    
        # bot spam tweet
        if(label == 's'):
            spam_usernames.append(dataset.loc[row_index, "Twitter @ Name"])
            dataset.drop(index=row_index, inplace=True)
            repeat = False

        if(repeat):
            print("Tag non riconosciuto.")

# adding spam users to users blacklist
users_blacklist_f = open("./Utils/usernames_blacklist.txt", "a")
for user in spam_usernames:
    users_blacklist_f.write(user)
    users_blacklist_f.write("\n")

# store tagged dataset into csv file
dataset.to_csv('./Dataset/samsung_week_5_to_12_October_tagged.csv')