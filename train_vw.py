# Train vowpal wabbit model for prediction of movie score

import pandas
import os
import sys
from vowpalwabbit import pyvw

# as always path to directory where this script is
pathToThisScript = os.path.abspath(os.path.dirname(sys.argv[0]))

training_df = pandas.read_csv(pathToThisScript + "/training_set.csv")

# print(training_df.head(10))

# configure vowpal wabbit no internal output, save cache and model, and added seed by hand
vw = pyvw.vw("--quiet -c -f betterFilmweb.vwmodel")
# data header originalTitle,isAdult,startYear,runtimeMinutes,averageRating,numVotes,directors,primaryName,deathYear


for i in training_df.index:
    rating = training_df.loc[i, 'averageRating']

    runtimeMinutes = training_df.loc[i, 'runtimeMinutes']
    isAdult = training_df.loc[i, 'isAdult']
    startYear = training_df.loc[i, 'startYear']
    numVotes = training_df.loc[i, 'numVotes']
    director = training_df.loc[i, 'primaryName']
    title = training_df.loc[i, 'originalTitle']
    
    runtimeMinutes = str(runtimeMinutes).replace("\\N", "0.0")
    director = str(director).replace(" ", "_")
    title = str(title).replace(" ", "_").replace(":", "-")

    learn_example = str(rating) + " | " + "runtimeMinutes:" + str(runtimeMinutes) + " isAdult:" + str(isAdult) + " startYear:" + str(startYear) + " numVotes:" + str(numVotes) + " " + str(director) + " " + str(title) 

    vw.learn(learn_example)

vw.finish()