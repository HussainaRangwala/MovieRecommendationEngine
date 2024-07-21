#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pyspark
import streamlit as st
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col, explode
import pandas as pd

# In[2]:


# Create a SparkSession
spark = SparkSession.builder     .appName("MovieRecommender")     .config("spark.executor.memory", "4g")     .config("spark.driver.memory", "4g")     .config("spark.network.timeout", "600s")     .config("spark.executor.heartbeatInterval", "60s")     .getOrCreate()

# Load movie ratings data from a file (replace with your data path)
ratings_df = spark.read.csv("Data/ratings.csv", header=True, inferSchema=True)
movies_df = spark.read.csv("Data/movies.csv", header=True, inferSchema=True)

# Show a sample of the data
ratings_df.show(5)
movies_df.show(5)


# In[2]:


# Create ALS model
als = ALS(maxIter=10, regParam=0.1, rank=10, userCol="userId", itemCol="movieId", ratingCol="rating", coldStartStrategy="drop")

# Split the data into training and test sets
(training, test) = ratings_df.randomSplit([0.8, 0.2])

# Train the model
model = als.fit(training)

# Evaluate the model by computing the RMSE on the test data
predictions = model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print(f"Root-mean-square error = {rmse}")


# In[20]:


# Get top 10 movie recommendations for a specific user
user_id = 123
user_recommendations = model.recommendForUserSubset(ratings_df.filter(col("userId") == user_id), 10)

# Explode the recommendations to get individual movie recommendations
exploded_recommendations = user_recommendations.select(explode("recommendations").alias("recommendation"))

# Extract recommended movie IDs as a DataFrame
recommended_movie_ids_df = exploded_recommendations.select(col("recommendation.movieId").alias("movieId"))

# Join with movies_df to get recommended movies
recommended_movies = recommended_movie_ids_df.join(movies_df, on="movieId", how="inner")

# Show the recommended movies
recommended_movies.show()


# In[12]:


# Initialize Spark session
spark = SparkSession.builder.appName("MovieRecommender").getOrCreate()

# Load data
ratings_df = spark.read.csv("Data/ratings.csv", header=True, inferSchema=True)
movies_df = spark.read.csv("Data/movies.csv", header=True, inferSchema=True)

# Train ALS model
als = ALS(maxIter=10, regParam=0.1, rank=10, userCol="userId", itemCol="movieId", ratingCol="rating", coldStartStrategy="drop")
model = als.fit(ratings_df)

# Streamlit GUI
st.image("background_img_2.jpg")
st.title("Movie Recommendation Engine")
user_id = st.number_input("Enter User ID", min_value=1, step=1)

if st.button("Get Recommendations"):
    user_recommendations = model.recommendForUserSubset(ratings_df.filter(col("userId") == user_id), 10)
    watched_history = ratings_df.filter(col("userId") == user_id)
    
    # Explode the recommendations to get individual movie recommendations
    exploded_recommendations = user_recommendations.select(explode("recommendations").alias("recommendation"))
    
    # Extract recommended movie IDs as a DataFrame
    recommended_movie_ids_df = exploded_recommendations.select(col("recommendation.movieId").alias("movieId"))
    
    # Join with movies_df to get recommended movies
    recommended_movies_list = recommended_movie_ids_df.join(movies_df, on="movieId", how="inner")
    
    # Collect the top 10 recommended movies
    recommended_movies_list = recommended_movies.limit(10).collect()
    
    # Convert the collected rows to a pandas DataFrame
    recommended_movies_pd_df = pd.DataFrame(recommended_movies_list, columns=["movieId", "title","genres"])
    
    # Display the DataFrame as a table in Streamlit
    st.write("Top 10 Movie Recommendations:")
    st.table(recommended_movies_pd_df)
    
    #Watched History
    watched_history = ratings_df.filter(col("userId") == user_id)
    watched_history_movie_ids_df = watched_history.select(col("movieId"))
    watched_movies =watched_history_movie_ids_df.join(movies_df, on="movieId", how="inner")
    
    watched_movies_list=watched_movies.collect()
    
    # Convert the collected rows to a pandas DataFrame
    watched_movies_pd_df = pd.DataFrame(watched_movies_list, columns=["movieId", "title","genres"])
    
    # Display the DataFrame as a table in Streamlit
    st.write("Watched History:")
    st.table(watched_movies_pd_df)

# Run Streamlit app
# In your terminal, run: streamlit run MovieRecommendationEngine.py

