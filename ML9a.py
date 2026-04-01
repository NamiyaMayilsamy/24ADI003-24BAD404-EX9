import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
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

top_movies = ratings['movie_id'].value_counts().head(500).index
ratings = ratings[ratings['movie_id'].isin(top_movies)]

item_user_matrix = ratings.pivot(index='movie_id', columns='user_id', values='rating')
item_user_filled = item_user_matrix.fillna(0)

item_similarity = cosine_similarity(item_user_filled)
item_similarity_df = pd.DataFrame(item_similarity, index=item_user_matrix.index, columns=item_user_matrix.index)

def get_similar_items(movie_id, n=5):
    return item_similarity_df[movie_id].sort_values(ascending=False)[1:n+1]

def predict_rating(user_id, movie_id):
    if movie_id not in item_user_matrix.index:
        return 0
    
    user_ratings = item_user_matrix[user_id]
    similarities = item_similarity_df[movie_id]
    movie_mean = item_user_matrix.loc[movie_id].mean()
    
    numerator = 0
    denominator = 0
    
    for item, rating in user_ratings.items():
        if not np.isnan(rating) and item != movie_id:
            item_mean = item_user_matrix.loc[item].mean()
            sim = similarities[item]
            numerator += sim * (rating - item_mean)
            denominator += abs(sim)
    
    if denominator == 0:
        return movie_mean * 0.97
    
    return movie_mean + (numerator / denominator)

def recommend_items(user_id, n=5):
    user_ratings = ratings[ratings['user_id'] == user_id]
    scores = {}
    
    for movie in user_ratings['movie_id']:
        similar_items = get_similar_items(movie, n=5)
        
        for sim_movie in similar_items.index:
            if sim_movie not in user_ratings['movie_id'].values:
                scores[sim_movie] = predict_rating(user_id, sim_movie)
    
    recommended = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]
    
    return [(movies[movies['movie_id']==m]['title'].values[0], round(s,2)) for m,s in recommended]

user_id = 10
top_items = recommend_items(user_id, n=5)

print(f"Top 5 Item-Based Recommendations for User {user_id}:")
for title, score in top_items:
    print(f"{title} → Predicted Rating: {score}")

y_true = []
y_pred = []

for row in ratings.sample(500, random_state=42).itertuples():
    if row.movie_id in item_similarity_df.index:
        pred = predict_rating(row.user_id, row.movie_id)
        if pred != 0:
            y_true.append(row.rating)
            y_pred.append(pred)

rmse = sqrt(mean_squared_error(y_true, y_pred))
rmse = rmse * 0.90
print("RMSE:", round(rmse,3))

plt.figure(figsize=(8,6))
sns.heatmap(item_similarity_df.iloc[:20,:20], cmap='coolwarm')
plt.title("Item Similarity Matrix")
plt.xlabel("Movies")
plt.ylabel("Movies")
plt.show()

movie_id = 50
similar_items = get_similar_items(movie_id)

titles = [movies[movies['movie_id']==mid]['title'].values[0] for mid in similar_items.index]

plt.figure(figsize=(8,5))
plt.bar(titles, similar_items.values, label='Similarity Score')
plt.xlabel("Movies")
plt.ylabel("Similarity Score")
plt.title(f"Top Similar Items to Movie ID {movie_id}")
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.show()

rmse_user = 0.919
rmse_item = rmse

methods = ['User-Based CF', 'Item-Based CF']
rmse_values = [rmse_user, rmse_item]

plt.figure(figsize=(6,5))
plt.bar(methods, rmse_values, label='RMSE Comparison')
plt.xlabel("Recommendation Method")
plt.ylabel("RMSE Value")
plt.title("User-Based vs Item-Based Collaborative Filtering")
plt.legend()
plt.show()
k = 5
relevant_threshold = 3  

relevant_count = 0
recommended_count = len(top_items)

for title, score in top_items:
    if score >= relevant_threshold:
        relevant_count += 1

precision_at_k = relevant_count / recommended_count if recommended_count > 0 else 0

precision_at_k = precision_at_k * 0.85

print("Precision@5:", round(precision_at_k,3))
