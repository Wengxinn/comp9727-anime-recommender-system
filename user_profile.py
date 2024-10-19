import helper
import random

# Get user preferences
class User_Profile():
    def __init__(self, preferred_genres=[], preferred_types=[], max_episodes=-1):
        self.preferred_genres = preferred_genres
        self.preferred_types = preferred_types
        self.max_episodes = max_episodes
        self.user_ratings = []
        self.user_profile = []
        self.recommended_animes = []

    def add_to_preferred_genres(self, genre):
        self.preferred_genres.append(genre)

    def remove_from_preferred_genres(self, genre):
        self.preferred_genres.remove(genre)

    def add_to_preferred_types(self, anime_type):
        self.preferred_types.append(anime_type)

    def remove_from_preferred_type(self, anime_type):
        self.preferred_types.remove(anime_type)

    def get_recommended_animes(self):
        return self.recommended_animes
    
    def add_to_recommended_animes(self, animes):
        self.recommended_animes += animes

    def update_max_episodes(self, max_episodes):
        self.max_episodes = max_episodes
    
    def cold_start_recommend(self, df, n=10):
        recs = []
        # No preferences
        if not self.preferred_genres and not self.preferred_types and self.max_episodes == -1:
            recs += helper.default_cold_start_recommendation(df, n_recs=n)
        else: 
            # Have preferred genres
            if self.preferred_genres: 
                recs += helper.get_animes_by_features(df, self.preferred_genres)
            
            # Have preferred types
            if self.preferred_types:
                recs += helper.get_animes_by_features(df, self.add_to_preferred_types)

            # Have preferred maximum number of episodes
            if self.max_episodes != -1:
                animes = helper.get_animes_by_episodes(df, self.max_episodes)
                recs += list(animes["anime_id"])

            # Add recommendations using the default settings 
            if len(recs) < n: 
                recs += helper.default_cold_start_recommendation(df, n_recs=(n-len(recs)))

            if len(recs) > n: 
                random.seed()
                recs = random.sample(recs, n)

        # Add the recommended animes to the user records
        self.add_to_recommended_animes(recs)
        return recs
    
    def get_user_ratings(self):
        return self.user_ratings
    
    def get_user_profile(self):
        return self.user_profile
    
    def add_to_user_ratings(self, df, animes):
        for anime in animes: 
            found = False
            for anime_rated in self.user_ratings:
                if anime_rated["id"] == anime["id"]: 
                    found = True 
            # Add to the user ratings if they are not yet been added
            if not found: 
                # Make sure the given anime id is in the dataset
                if helper.get_animes_by_ids(df, [anime["id"]]):
                    self.user_ratings.append(anime)

    def cold_start(self, df, n=10):
        # Generate more recommendations with the user watched less than 10 animes 
        while len(self.user_ratings) < n:
            recs = self.cold_start_recommend(df, n)

            random.seed(42)
            animes_rated = []
            for rec in recs: 
                user_rated = random.choice([True, False])
                if user_rated: 
                    rating = random.randint(1, 10)
                    anime = {}
                    anime["id"] = rec
                    anime["rating"] = rating
                    animes_rated.append(anime)
                    
            self.add_to_user_ratings(df, animes_rated)  

    def build_user_profile(self, minimum_rating):
        animes_liked = [anime_rating["id"] for anime_rating in self.user_ratings if anime_rating["rating"] >= minimum_rating]
        self.user_profile += animes_liked

    def get_possible_recommendations(self, df):
        possible_recs = df.loc[~df["anime_id"].isin(self.recommended_animes)]
        return possible_recs

    def random_content_based_recommend(self, df, minimum_rating, n=50):
        while len(self.user_profile) < n:
            animes_liked = df.loc[df["anime_id"].isin(self.user_profile)]
            animes_liked = helper.get_normalised_df(animes_liked)

            possible_recs = self.get_possible_recommendations(df)
            possible_recs_id = list(possible_recs["anime_id"])
            possible_recs = helper.get_normalised_df(possible_recs)

            recs = helper.content_based_recommend(n, animes_liked, possible_recs, possible_recs_id)
            
            random.seed(42)
            animes_rated = []
            for rec in recs: 
                user_rated = random.choice([True, False])
                if user_rated: 
                    rating = random.randint(1, 10)
                    anime = {}
                    anime["id"] = rec
                    anime["rating"] = rating
                    animes_rated.append(anime)
                    
            self.add_to_user_ratings(df, animes_rated) 
            self.build_user_profile(minimum_rating)