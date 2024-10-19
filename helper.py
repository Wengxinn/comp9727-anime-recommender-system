"""
Helper functions for comp9727 project recommender system.
"""
# Importing libraries
import pandas as pd
import random

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

def vectorise(data, cv=True, max_features=None):
    """
    Convert the given text data into numerical data by either CountVectorizer or TfidVectorizer. 
    """
    # Count vectoriser
    # Convert text data into token counts (count of occurrence in the document)
    if cv: 
        vectorizer = CountVectorizer(max_features=max_features)
    # Tfidf vectoriser
    # Convert text data into tfidf weights (measure how relevant the words to the document)
    else: 
        vectorizer = TfidfVectorizer(max_features=max_features)
    # Return value of x: (document_id, token_id), token counts/tfidf weights
    x = vectorizer.fit_transform(data)
    
    x_feature_names = vectorizer.get_feature_names_out()
    return x, x_feature_names

def document_term_matrix(vectoriser_output, words):
    """
    Convert the vectoriser output into a document term matrix. 
    """
    # Document term matrix: document (row), token (column)
    return pd.DataFrame(vectoriser_output.toarray(), columns=list(words))

def remove_separator(text, separator=", "):
    """
    Remove the separator between words in the given text.
    """
    text = text.split(separator)
    return " ".join(text)

def topn_animes_by_column(df, column, n):
    """
    Returna list of top-n animes sorted by the specified column. 
    """
    sorted_animes = df.sort_values(by=column, ascending=False)
    topn_animes = sorted_animes["anime_id"].tolist()[:n]
    return topn_animes

def get_animes_by_ids(df, ids):
    """
    Return a list of animes corresponds to the given list of ids. 
    """
    new_df = df[df["anime_id"].isin(ids)]
    return new_df["name"].tolist()

def get_animes_by_feature(df, feature):
    """
    Return a dataframe with animes that are categorised under the specified feature.
    """
    animes = df.loc[(df[feature] == 1)]
    return animes

def topn_genres(genre_df, n):
    """
    Return a dictionary of top-n genres with their counts.
    """
    genre_counts = genre_df.sum()
    genre_counts_tup = (zip(genre_df.columns, genre_counts))
    genre_counts_tup = sorted(genre_counts_tup, key=lambda x: x[1], reverse=True)[:n]
    genre_counts_dict = dict(genre_counts_tup)
    return genre_counts_dict

def topn_highest_rating_anime_by_feature(df, feature, n):
    """
    Return a dataframe with top-n highest-rated animes categorised under the specified feature. 
    """
    animes_by_feature = get_animes_by_feature(df, feature)
    topn_highest_rated = topn_animes_by_column(animes_by_feature, "rating", n)
    topn_highest_rated = get_animes_by_ids(df, topn_highest_rated)
    return topn_highest_rated

def topn_most_popular_anime_by_feature(df, feature, n):
    """
    Return a dataframe with top-n most popular animes categorised under the specified feature. 
    """
    animes_by_feature = get_animes_by_feature(df, feature)
    topn_popular = topn_animes_by_column(animes_by_feature, "members", n)
    topn_popular = get_animes_by_ids(df, topn_popular)
    return topn_popular

def normalised_data(df, drop_columns):
    """
    Normalise the given dataframe, without the values in drop_columns.
    """
    scaler = MinMaxScaler()
    normalised_data = scaler.fit_transform(df.drop(drop_columns, axis=1))
    normalised_df = pd.DataFrame(normalised_data)
    normalised_df = pd.concat([df[drop_columns], normalised_df], axis=1)
    normalised_df.columns = df.columns
    return normalised_df

def get_animes_by_episodes(df, max_episodes):
    """
    Return a dataframe with animes that has episodes no more than the given max_episodes.
    """
    animes = df.loc[(df["episodes"] <= max_episodes)]
    return animes

def recommend_topn_high_rated_genres(df, n, minimum_rating):
    """
    Return list of anime ids that are categorised under the top-n genres and have rating no less than the specified minimum rating.
    """
    # Get genres columns 
    genre_df = df.iloc[:, 5:52]
    top_n_genres = topn_genres(genre_df, n)
    # Get animes that are categorised under these genres filter out animes that have rating less than the minimum rating
    ids = get_animes_by_features(df.loc[(df["rating"] >= minimum_rating)], top_n_genres)
    return ids

def get_animes_by_features(df, features):
    """
    Select all animies based on the feature list and return their corresponding anime ids. 
    """
    ids = []
    for feature in features: 
        animes = get_animes_by_feature(df, feature)
        ids += animes["anime_id"].tolist()
    ids = list(set(ids))
    return ids
    
def default_cold_start_recommendation(df, n_recs=10, n_genres=10, minimum_rating=8, n_sample=10):
    """
    Generate recommendation for cold-start situation. 
    """
    random.seed(1)

    # Top-n genres animes
    # Select animes that are categorised under the top-(n_genres) genres and have rating >= minimum_rating
    top_n_genres_animes = recommend_topn_high_rated_genres(df, n_genres, minimum_rating)
    top_n_genres_animes = df.loc[df["anime_id"].isin(top_n_genres_animes)]
    # Randomly sample n_sample animes from the generated animes 
    top_n_genres_animes_sample = top_n_genres_animes.sample(n=min(n_sample, len(top_n_genres_animes)))

    # Most popular animes
    top_n_most_popular_animes = topn_animes_by_column(df, "members", n_sample)
    top_n_most_popular_animes = df.loc[df["anime_id"].isin(top_n_most_popular_animes)]
    top_n_most_popular_animes_sample = top_n_most_popular_animes.sample(n=min(n_sample, len(top_n_most_popular_animes)))
    
    # Combine both generated recommended animes and drop duplicates, then randomly select n_recs recommendations
    recs_df = pd.concat([top_n_genres_animes_sample, top_n_most_popular_animes_sample])
    recs_df.drop_duplicates()
    recs_df.reset_index(drop=True)
    recs_df = recs_df.sample(n=min(n_recs, len(recs_df)))
    return list(recs_df["anime_id"])

def get_normalised_df(df):
    """
    Return a dataframe with only normalised columns: all columns except anime_id, name, episodes, rating and members.
    """
    return df.drop(["anime_id", "name", "episodes", "rating", "members"], axis=1)

def content_based_recommend(n_recs, user_liked, possible_recs, possible_recs_id):
    cosim_user = cosine_similarity(user_liked, possible_recs)
    cosim_sum = cosim_user.sum(axis=0)

    cosim_dict = {}
    for i in range(len(cosim_sum)):
        cosim_dict[possible_recs_id[i]] = cosim_sum[i]

    sorted_cosim = sorted(cosim_dict.items(), key=lambda x: x[1], reverse=True)
    animes = [anime_id for (anime_id, cosim) in sorted_cosim]
    recs_id = animes[:n_recs]
    return recs_id