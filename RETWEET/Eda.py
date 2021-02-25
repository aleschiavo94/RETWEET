import re 
import pandas as pd 

# loading input data
dataset = pd.read_csv("./Dataset/merged_weeks_1_2_3_4.csv")

print(dataset.head(5))


print("Dataset loaded", "\n")
print("Numero righe: {}" .format(len(dataset.index)))

neg = len(dataset[dataset['tag'] == 'negative'])
pos = len(dataset[dataset['tag'] == 'positive'])
neu = len(dataset[dataset['tag'] == 'neutral'])

print("# of positive samples: {}".format(pos))
print("# of negative samples: {}".format(neg))
print("# of neutral samples: {}".format(neu))

tot = neg + pos + neu
print("total # of samples: {}".format(tot))

