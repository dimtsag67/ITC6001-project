import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import json


# Function that determines z-value outliers on a given series
def z_score_outlier_detection(data, thresh):
    outliers = []
    mean = np.mean(data)
    std = np.std(data)
    for point in data:
        z = (point - mean) / std
        if z > thresh:
            outliers.append(point)
    return outliers, len(outliers)
    # print('outliers in dataset are', outliers)


# Function using similarity values to determine k-nearest neighbors
def find_n_neighbours(df, n):
    top_neighbours = df.apply(lambda x: pd.Series(x.sort_values(ascending=False)
                                                  .iloc[1:n + 1].index,
                                                  index=['Neighbour{}'.format(i) for i in range(1, n + 1)]), axis=1)
    return top_neighbours


# Function predicting the rating of a user towards a book through collaborative filtering
def predict(userId, itemId, ratings, similar_users, similarities):
    # number of neighbours to consider
    ncolumns = similar_users.shape[1]

    sum = 0.0
    similar_sum = 0.0
    for k in range(0, ncolumns):
        neighbor = int(similar_users.loc[userId][k])
        # weighted sum
        sum = sum + ratings.loc[neighbor][itemId] * similarities[neighbor][userId]
        similar_sum = similar_sum + similarities[neighbor][userId]
    return sum / similar_sum


# Loading a portion of the large files
df_Books = pd.read_csv('BX-Books.csv', header=0, nrows=300000, delimiter=';', encoding='ISO-8859-1')
df_Ratings = pd.read_csv('BX-Book-Ratings.csv', header=0, nrows=10000000, delimiter=";", encoding='ISO-8859-1')
df_Users = pd.read_csv('BX-Users.csv', header=0, nrows=300000, delimiter=';', encoding='ISO-8859-1')

# Data pre-processing


# Lower casing and replacing '-' with '_' for easier access and indexing
df_Users.columns = df_Users.columns.str.strip().str.lower().str.replace('-', '_')
df_Ratings.columns = df_Ratings.columns.str.strip().str.lower().str.replace('-', '_')
df_Books.columns = df_Books.columns.str.strip().str.lower().str.replace('-', '_')

# Some ages were 0 were replaced by Nan values
df_Users['age'].fillna(0, inplace=True)
df_Users['age'] = df_Users['age'].astype(int)

# Casting ISBN feature to string type
df_Books['isbn'] = df_Books['isbn'].astype(str)
df_Ratings['isbn'] = df_Ratings['isbn'].astype('string')

# Dropping the image URLs of the books as we dont need them
df_Books.drop(df_Books.iloc[:, 5:8], inplace=True, axis=1)

# Removing User ratings from users not existent in Users table
df_Ratings = df_Ratings[df_Ratings.user_id.isin(df_Users.user_id)]

# Removing implicit ratings with a value of 0.
df_Ratings = df_Ratings[df_Ratings.book_rating != 0]

# Merge Ratings with books and users so to derive popular books/authors/ages.
df3 = pd.merge(df_Ratings, df_Books, on='isbn')
df4 = pd.merge(df_Ratings, df_Users, on='user_id')

Authors_activity = df3.book_author.value_counts()
Books_reading_activity = df3.book_title.value_counts()
UserAges_reading_activity = df4.age.value_counts()[1:]

# Outlier Detection using z-score with a threshold of 3

threshold = 3
Author_outliers = z_score_outlier_detection(Authors_activity, threshold)
Book_outliers = z_score_outlier_detection(Books_reading_activity, threshold)
User_outliers = z_score_outlier_detection(UserAges_reading_activity, threshold)

print("Authors being read outliers :", Author_outliers[1])
print("Books being read outliers :", Book_outliers[1])
print("Users that rated outliers :", User_outliers[1])

# POPULARITY OF BOOKS ,AUTHORS AND AGES BY READING ACTIVITY
plt1 = Authors_activity[:10].plot(kind='bar', title='Top 10 Popularity Authors',
                                  yticks=[1000, 2000, 3000, 4000, 5000])
plt.show()

plt2 = Books_reading_activity[:10].plot(kind='barh', title='Top 10 Popularity Books',
                                        yticks=[100, 200, 300, 400])
plt.show()

plt3 = UserAges_reading_activity[:10].plot(kind='bar', title='TOP 10 Reading active Ages',
                                           yticks=[3500, 5500, 7500, 9500, 11500, 13500])

plt.show()

# Removing low popularity book records
Low_Books_count = df_Ratings['isbn'].value_counts()
df_Ratings = df_Ratings[df_Ratings.isin(Low_Books_count.index[Low_Books_count > 10]).values]

# Removing low user-activity records
Low_User_count = df_Ratings['user_id'].value_counts()
df_Ratings = df_Ratings[df_Ratings.isin(Low_User_count.index[Low_User_count > 10]).values]

# Removing duplicates
df_Ratings.drop_duplicates(inplace=True)

# Transforming our Ratings table
Ratings_transformed = pd.pivot_table(df_Ratings, columns='isbn', index='user_id', values='book_rating')
Ratings_transformed.fillna(0, inplace=True)

# Applying cosine similarity so to get user-pair similarity values
cosine_correlation = cosine_similarity(Ratings_transformed)

# Temporarily changing diagonal to 0 so to get the highest user-pair value
np.fill_diagonal(cosine_correlation, 0)
print("Largest Cosine correlation User-pair value:", cosine_correlation.max())
np.fill_diagonal(cosine_correlation, 1)

cosine_similarities = pd.DataFrame(cosine_correlation, index=Ratings_transformed.index)
cosine_similarities.columns = Ratings_transformed.index

cosine_similarities.to_csv('user-pairs-books.data')

# Obtain the k-nearest neighbors
no_of_neighbours = 5
similar_users = find_n_neighbours(cosine_similarities, no_of_neighbours)

neighbors_dict = similar_users.to_dict(orient='index')

with open('neighbors-k-books.data', 'w') as file:
    json.dump(neighbors_dict, file)


# Predicting and evaluating the ratings of books for each user
nUsers = list(df_Ratings.user_id.unique())
nItems = list(df_Ratings.isbn.unique())

error = 0
cnt = 0
for i in nUsers[:100]:
    for j in nItems[:100]:
        prediction = predict(i, j, Ratings_transformed, similar_users, cosine_similarities)
        error = error + np.abs(prediction - Ratings_transformed._get_value(i, j))
        cnt += 1

Mean_absolute_error = error/cnt

# Alternative method for taking pairs of users through a dictionary
# for i in df_Ratings['user_id'].unique():
# d[i] = [{df_Ratings['isbn'][j]: df_Ratings['book_rating'][j]} for j in df_Ratings[df_Ratings['user_id'] == i].index]
