# Train vowpal wabbit model for prediction of movie score

import pandas
import os
import sys
from vowpalwabbit import pyvw

# as always path to directory where this script is
pathToThisScript = os.path.abspath(os.path.dirname(sys.argv[0]))

improve_df = pandas.read_csv(pathToThisScript + "/improve_set.csv")

print(improve_df.head(10))

# configure vowpal wabbit
vw = pyvw.vw("--quiet -i betterFilmweb.vwmodel -f improvedBetterFilmweb.vwmodel -c --passes 4 --holdout_off --hash all --loss_function squared -l 2 --random_seed 32167") #  --passes 4 --holdout_off --noconstant --hash all --bfgs --ftrl --loss_function squared -l 0.4 --l2


for i in improve_df.index:
    rating = improve_df.loc[i, 'averageRating']

    runtimeMinutes = improve_df.loc[i, 'runtimeMinutes']
    isAdult = improve_df.loc[i, 'isAdult']
    startYear = improve_df.loc[i, 'startYear']
    numVotes = improve_df.loc[i, 'numVotes']
    director = improve_df.loc[i, 'primaryName']
    title = improve_df.loc[i, 'originalTitle']
    
    runtimeMinutes = str(runtimeMinutes).replace("\\N", "0.0")
    director = str(director).replace(" ", "_")
    title = str(title).replace(" ", "_").replace(":", "-")

    learn_example = str(rating) + " | " + "runtimeMinutes:" + str(runtimeMinutes) + " isAdult:" + str(isAdult) + " startYear:" + str(startYear) + " numVotes:" + str(numVotes) + " " + str(director) + " " + str(title) 

    vw.learn(learn_example)

vw.finish()