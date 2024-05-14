#import all the required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Define the file path as a variable
file_path = 'songs_normalize.csv'

# Import and read the dataset
top_songs = pd.read_csv(file_path)

# Checking the first rows of data
top_songs.head()

print(top_songs.head()) 

# Basic descriptive statistics
top_songs.describe()

print(top_songs.describe())

# Collecting the total number of observations in the database
total_rows = len(top_songs)
print('Total of observations:', total_rows)

# Calculating the number of missing values
missing_values = top_songs.isnull().sum()
total_missing = missing_values.sum()

# Calculating the proportion between total missing values and total observations of the data
proportion_missing_values = (total_missing/total_rows)*100

print(missing_values)
print('-- Proportion of missing values:', proportion_missing_values)


# Checking the variables format
top_songs.info()


print(top_songs.info())


# Creating a function that converts milliseconds to seconds
def convert_ms_to_sec(ms):
    return ms/1000

# Creating a new column in the dataframe, using the apply function with lambda
top_songs['duration_sec'] = top_songs.apply(lambda ms: convert_ms_to_sec(ms['duration_ms']), axis=1)


# Converting the new column to 'int64' and checking for missing values in it
top_songs = top_songs.astype({"duration_sec": np.int64})
top_songs.info()

print(top_songs.info())


#size() funciton : pulls up the unique groupby count 
#reset_index(): method that resets the name of the column 
#to_frame(): method that convert the GroupBy into a dataframe

songs_by_year = top_songs.groupby("year").size().to_frame(name = 'songs').reset_index()
songs_by_year


# Calculating the average of observations by year
print("Average of songs by year:",songs_by_year['songs'].mean(axis = 0))

# Droping the 1998, 1999 and 2020 observations
top_songs = top_songs[(top_songs.year >= 2000) & (top_songs.year <= 2019)]

# Checking if the "year" is in the right range
print(top_songs.groupby("year")["song"].count())


# First let's group and count all music genres in the dataset and get the top 10
songs_by_genre = top_songs \
                    .groupby("genre") \
                    .size().to_frame(name = 'songs').reset_index() \
                    .sort_values(['songs'],ascending=False).head(10)

print(songs_by_genre)


#below code starts for horizontal bar chart

# plotting a bar chart
plt.figure(figsize = (10,7))
plots = sns.barplot(x = 'songs', y = 'genre', data = songs_by_genre, palette = 'rocket')  

# naming the x-axis and the y-axis
plt.xlabel('Number of songs', size = 13)
plt.ylabel('Genre', size = 13)
  
# title of the graph
plt.title('Number of songs by genre (2000 - 2019)', size = 18)
  

# Creating a dataframe with the number of each genre per year (and convertig 'year' as time variable)
songs_by_genre_year = top_songs[['year','genre']] \
                    .groupby(['year', 'genre']) \
                    .size().to_frame(name = 'songs').reset_index()

songs_by_genre_year['year'] = songs_by_genre_year['year'].astype(int)


# Inner join to limit the genres to ones on the top 10
songs_by_genre_year = pd.merge(songs_by_genre_year, songs_by_genre['genre'], \
                               on = 'genre', \
                               how='inner')



genre_pop = songs_by_genre_year[(songs_by_genre_year.genre == 'pop')]
genre_hiphop_pop = songs_by_genre_year[(songs_by_genre_year.genre == 'hip hop, pop')]
genre_hipop_pop_rb = songs_by_genre_year[(songs_by_genre_year.genre == 'hip hop, pop, R&B')]
genre_pop_elec = songs_by_genre_year[(songs_by_genre_year.genre == 'pop, Dance/Electronic')]
genre_pop_rb = songs_by_genre_year[(songs_by_genre_year.genre == 'pop, R&B')]


plt.subplots(figsize=(17, 8),sharey=True)


# using subplot function and creating plot one
ax1 = plt.subplot(3,2,1)  # row 1, column 2, count 1

z = np.polyfit(genre_pop['year'], genre_pop['songs'], 1) #Polynomial fit
p = np.poly1d(z)

plt.plot(genre_pop['year'], genre_pop['songs'], 'c', linewidth=1)
plt.plot(genre_pop['year'],p(genre_pop['year']),"r:")
plt.title('Number of Pop songs (2000 - 2019)')
plt.xlabel('Year')
plt.ylabel('Number of top songs')
 

ax2 = plt.subplot(3, 2, 2)

z = np.polyfit(genre_hiphop_pop['year'], genre_hiphop_pop['songs'], 1) #Polynomial fit
p = np.poly1d(z)

plt.plot(genre_hiphop_pop['year'], genre_hiphop_pop['songs'], 'g', linewidth=1)
plt.plot(genre_hiphop_pop['year'],p(genre_hiphop_pop['year']),"r:")
plt.title('Number of Hip-hop / Pop songs (2000 - 2019)')
plt.xlabel('Year')
plt.ylabel('Number of top songs')

ax1.sharey(ax2)

ax3 = plt.subplot(3,2,3)

z = np.polyfit(genre_hipop_pop_rb['year'], genre_hipop_pop_rb['songs'], 1) #Polynomial fit
p = np.poly1d(z)

plt.plot(genre_hipop_pop_rb['year'], genre_hipop_pop_rb['songs'], 'b', linewidth=1)
plt.plot(genre_hipop_pop_rb['year'],p(genre_hipop_pop_rb['year']),"r:")
plt.title('Number of Hip-hop / Pop / R&B songs (2000 - 2019)')
plt.xlabel('Year')
plt.ylabel('Number of top songs')

#ax1.sharey(ax3)

ax4 = plt.subplot(3, 2, 4)

z = np.polyfit(genre_pop_elec['year'], genre_pop_elec['songs'], 1) #Polynomial fit
p = np.poly1d(z)

plt.plot(genre_pop_elec['year'], genre_pop_elec['songs'], 'y', linewidth=1)
plt.plot(genre_pop_elec['year'],p(genre_pop_elec['year']),"r:")
plt.title('Number of Pop / Dance / Eletronic songs (2000 - 2019)')
plt.xlabel('Year')
plt.ylabel('Number of top songs')

#ax1.sharey(ax4)

ax4 = plt.subplot(3, 2, 5)

z = np.polyfit(genre_pop_rb['year'], genre_pop_rb['songs'], 1) #Polynomial fit
p = np.poly1d(z)

plt.plot(genre_pop_rb['year'], genre_pop_rb['songs'], 'y', linewidth=1)
plt.plot(genre_pop_rb['year'],p(genre_pop_rb['year']),"r:")
plt.title('Number of Pop / R&B songs (2000 - 2019)')
plt.xlabel('Year')
plt.ylabel('Number of top songs')



# space between the plots
plt.tight_layout()
 
# show plot
plt.show()

