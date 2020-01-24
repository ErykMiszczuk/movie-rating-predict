# This script prepare data for vowpal wabbit.
# Input: imdb data
# Output: filtered movie data, using only films from 50 years ago

import pandas
import numbers
import os
import sys

# as always path to directory where this script is
pathToThisScript = os.path.abspath(os.path.dirname(sys.argv[0]))
# we need to delete repeated data in all imported data sources because imdb export tsv contains duplicates (dunno why)

# loading basic data
title_basic_data = pandas.read_table(pathToThisScript + '/imdb_datasets/title.basics.tsv', sep='\t')

# selecting only movies since 1970
# drop unused columns and get only movies
# title_basic_data = title_basic_data[['tconst', 'titleType', 'primaryTitle', 'originalTitle', 'startYear']].query('titleType == \"movie\"')
# print(title_basic_data.tail(10))
# drop movies with no start date
# title_basic_data = title_basic_data.loc[title_basic_data['startYear'] != r'\N']
# print(title_basic_data.tail(10))
# drop movies before 1970
# title_basic_data = title_basic_data.astype({'startYear': 'int16'}).query(('titleType == movie') and ('startYear > 1970'))
# attempt to do whole thing in one method chain
title_basic_data = title_basic_data.drop(columns = ['genres', 'endYear']).query('titleType == \"movie\"').loc[title_basic_data['startYear'] != r'\N'].astype({'startYear': 'int16'}).query(('titleType == movie') and ('startYear > 1970'))
# print(title_basic_data.head(10))
# well it works like a charm!

# load ratings data
title_score_data = pandas.read_table(pathToThisScript + '/imdb_datasets/title.ratings.tsv', sep='\t')
# print(title_score_data.head(10))

# now we need to join these two tables
title_data = pandas.merge(title_basic_data, title_score_data, on='tconst', how='left').drop_duplicates()
# print(title_data.head(10))
# nice we have score for film but we still lack director data

# load directors data
directors_data = pandas.read_table(pathToThisScript + '/imdb_datasets/title.crew.tsv', sep='\t')
# print(directors_data.head(10))
# another join of tables
title_data = pandas.merge(title_data, directors_data, on='tconst', how='left').drop_duplicates()

# load directors names
names_data = pandas.read_table(pathToThisScript + '/imdb_datasets/name.basics.tsv', sep='\t')
# drop unnecessary columns from names data
names_data = names_data.drop(columns=['birthYear', 'primaryProfession', 'knownForTitles'])
# print(names_data.head(10))
# now add directors data to movies
title_data = pandas.merge(title_data, names_data, left_on='directors', right_on='nconst', how='left').drop_duplicates()

# so right now Ive got what i needed with small additions so again drop unused columns
title_data = title_data.drop(columns=['writers', 'nconst', 'primaryTitle', 'titleType'])

title_data = title_data.loc[title_data['averageRating'].notna()]
title_data = title_data.loc[title_data['primaryName'].notna()]

print(title_data.head(10))
# if title_basic_data:
#     print("Basics data loaded")

# now thanks to pandas create 3 sets: training set, improve set, test set
training_set = title_data.sample(frac=0.7)

rest = title_data.loc[~title_data.index.isin(training_set)]

improve_set = rest.sample(frac=0.15)

test_set = rest.loc[~rest.index.isin(improve_set)]

training_set.to_csv(pathToThisScript + '/training_set.csv')
improve_set.to_csv(pathToThisScript + '/improve_set.csv')
test_set.to_csv(pathToThisScript + '/test_set.csv')