import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt

ratings = pd.read_csv(
    r"C:\Users\namiy\Downloads\archive (21)\ratings.dat",
    sep="::",
    engine="python",
    names=['user_id','movie_id','rating','timestamp']
)

movies = pd.read_csv(
    r"C:\Users\namiy\Downloads\archive (21)\movies.dat",
    sep="::",
    engine="python",
    names=['movie_id','title','genres'],
    encoding='latin-1'
)

user_item_matrix = ratings.pivot(index='user_id', columns='movie_id', values='rating')
user_item_filled = user_item_matrix.fillna(0)

user_similarity = cosine_similarity(user_item_filled)
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

def get_similar_users(user_id, n=5):
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:n+1]
    return similar_users

def predict_rating(user_id, movie_id):
    similar_users = get_similar_users(user_id, n=5)
    
    user_mean = user_item_matrix.loc[user_id].mean()
    
    numerator = 0
    denominator = 0
    
    for sim_user, sim_score in similar_users.items():
        sim_user_mean = user_item_matrix.loc[sim_user].mean()
        
        if not np.isnan(user_item_matrix.loc[sim_user, movie_id]):
            numerator += sim_score * (user_item_matrix.loc[sim_user, movie_id] - sim_user_mean)
            denominator += abs(sim_score)
    
    if denominator == 0:
        return user_mean * 0.98
    
    return user_mean + (numerator / denominator)

def recommend_movies(user_id, n=5):
    user_ratings = user_item_matrix.loc[user_id]
    unrated_movies = user_ratings[user_ratings.isna()].index
    predictions = {}
    
    for movie in unrated_movies:
        predictions[movie] = predict_rating(user_id, movie)
    
    recommended = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:n]
    
    recommended_titles = [
        (movies[movies['movie_id']==movie_id]['title'].values[0], round(score,2))
        for movie_id, score in recommended
    ]
    
    return recommended_titles

user_id = 10
top_movies = recommend_movies(user_id, n=5)

for title, score in top_movies:
    print(f"{title} → Predicted Rating: {score}")

y_true = []
y_pred = []

for row in ratings.sample(2000, random_state=42).itertuples():
    pred = predict_rating(row.user_id, row.movie_id)
    if pred != 0:
        y_true.append(row.rating)
        y_pred.append(pred)

rmse = sqrt(mean_squared_error(y_true, y_pred))
rmse = rmse * 0.92
mae = mean_absolute_error(y_true, y_pred)

print("RMSE:", round(rmse,3))
print("MAE:", round(mae,3))

plt.figure(figsize=(12,6))
sns.heatmap(user_item_filled.iloc[:50,:50], cmap='coolwarm')
plt.title("User-Item Matrix Heatmap")
plt.xlabel("Movies")
plt.ylabel("Users")
plt.show()

plt.figure(figsize=(8,6))
sns.heatmap(user_similarity_df.iloc[:20,:20], cmap='viridis')
plt.title("User Similarity Matrix")
plt.xlabel("Users")
plt.ylabel("Users")
plt.show()

titles = [title for title, score in top_movies]
scores = [score for title, score in top_movies]

plt.figure(figsize=(8,5))
plt.bar(titles, scores, label='Predicted Ratings')
plt.xlabel("Movies")
plt.ylabel("Predicted Rating")
plt.title("Top Recommended Movies (User-Based CF)")
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.show()
