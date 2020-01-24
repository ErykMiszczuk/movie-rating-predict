# Testing vowpal wabbit model in comparison to real data

import pandas
import os
import sys
import re
from math import sqrt, pow
from vowpalwabbit import pyvw

# as always path to directory where this script is
pathToThisScript = os.path.abspath(os.path.dirname(sys.argv[0]))

# pattern to get score predicted by model from output
rating_pattern = re.compile(r"^\d.[\d]*")

test_df = pandas.read_csv(pathToThisScript + "/test_set.csv")

# configure vowpal wabbit to load model
vw = pyvw.vw("--quiet -i betterFilmweb.vwmodel")
# data header originalTitle,isAdult,startYear,runtimeMinutes,averageRating,numVotes,directors,primaryName,deathYear

# list containing ratings predicted by model
predicted_ratings = []
# list of actual ratings
actual_ratings = []

for i in test_df.index:
    rating = test_df.loc[i, 'averageRating']
    actual_ratings.append(float(rating))

    runtimeMinutes = test_df.loc[i, 'runtimeMinutes']
    isAdult = test_df.loc[i, 'isAdult']
    startYear = test_df.loc[i, 'startYear']
    numVotes = test_df.loc[i, 'numVotes']
    director = test_df.loc[i, 'primaryName']
    title = test_df.loc[i, 'originalTitle']
    
    runtimeMinutes = str(runtimeMinutes).replace("\\N", "0.0")
    director = str(director).replace(" ", "_")
    title = str(title).replace(" ", "_").replace(":", "-")

    test_example = "| " + "runtimeMinutes:" + str(runtimeMinutes) + " isAdult:" + str(isAdult) + " startYear:" + str(startYear) + " numVotes:" + str(numVotes) + " " + str(director) + " " + str(title) 

    prediction = str(vw.predict(test_example))
    # print(prediction)
    predicted_ratings.append(prediction)

num_of_samples = len(actual_ratings)

# print(actual_ratings)
# print(predicted_ratings)

sum_rsme = 0
# (zfi - zoi)^2 / num_of_samples
for i, val in enumerate(actual_ratings):
    f = float(predicted_ratings[i])
    o = float(val)
    sum_rsme += (pow(f - o, 2) / num_of_samples)

print("RMSE in default model: {}".format(sqrt(sum_rsme)))

vw.finish()

# RMSE for improve set

vw = pyvw.vw("--quiet -i improvedBetterFilmweb.vwmodel")
# data header originalTitle,isAdult,startYear,runtimeMinutes,averageRating,numVotes,directors,primaryName,deathYear

# list containing ratings predicted by model
predicted_ratings = []
# list of actual ratings
actual_ratings = []

for i in test_df.index:
    rating = test_df.loc[i, 'averageRating']
    actual_ratings.append(float(rating))

    runtimeMinutes = test_df.loc[i, 'runtimeMinutes']
    isAdult = test_df.loc[i, 'isAdult']
    startYear = test_df.loc[i, 'startYear']
    numVotes = test_df.loc[i, 'numVotes']
    director = test_df.loc[i, 'primaryName']
    title = test_df.loc[i, 'originalTitle']
    
    runtimeMinutes = str(runtimeMinutes).replace("\\N", "0.0")
    director = str(director).replace(" ", "_")
    title = str(title).replace(" ", "_").replace(":", "-")

    test_example = "| " + "runtimeMinutes:" + str(runtimeMinutes) + " isAdult:" + str(isAdult) + " startYear:" + str(startYear) + " numVotes:" + str(numVotes) + " " + str(director) + " " + str(title) 

    prediction = str(vw.predict(test_example))
    # print(prediction)
    predicted_ratings.append(prediction)

num_of_samples = len(actual_ratings)

# print(actual_ratings)
# print(predicted_ratings)

sum_rsme = 0
# (zfi - zoi)^2 / num_of_samples
for i, val in enumerate(actual_ratings):
    f = float(predicted_ratings[i])
    o = float(val)
    sum_rsme += (pow(f - o, 2) / num_of_samples)

print("RMSE in improved model: {}".format(sqrt(sum_rsme)))

vw.finish()