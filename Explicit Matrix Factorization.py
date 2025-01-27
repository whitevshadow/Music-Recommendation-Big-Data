from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.sql.functions import col, lit, explode
from pyspark.sql.types import StringType
import os
import time
from pyspark.ml.linalg import Vectors
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

# Set Spark Local IP if needed
os.environ['SPARK_LOCAL_IP'] = '192.168.0.103'

def recommend_music(song_name, song_data_path, user_data_path, num_recommendations=5):
    # Create a Spark session
    spark = SparkSession.builder \
        .appName("Music Recommendation System") \
        .config("spark.executor.memory", "8g") \
        .config("spark.ui.port", "4057") \
        .config("spark.ui.enabled", "false") \
        .config("spark.driver.memory", "8g") \
        .config("spark.driver.bindAddress", "127.0.0.1") \
        .config("spark.driver.host", "127.0.0.1") \
        .getOrCreate()

    # Set logging level to suppress warnings
    spark.sparkContext.setLogLevel("ERROR")

    start_time = time.time()

    # Load the JSON files
    song_data_df = spark.read.json(song_data_path)
    user_data_df = spark.read.json(user_data_path)

    # Ensure song_id is treated as a string
    song_data_df = song_data_df.withColumn("song_id", col("song_id").cast(StringType()))

    # Flatten the user_data_df if songs column is an array of structs
    if user_data_df.schema["songs"].dataType.typeName() == "array":
        user_data_df = user_data_df.withColumn("song", explode(col("songs"))).select("user_id",
                                                                                     col("song.song_id").alias(
                                                                                         "song_id"))

    # Create triplets_df with listen_count
    triplets_df = user_data_df.withColumn("listen_count", lit(1))

    # Convert user_id and song_id to numeric IDs, handle unseen labels
    user_indexer = StringIndexer(inputCol="user_id", outputCol="user_index", handleInvalid="keep")
    song_indexer = StringIndexer(inputCol="song_id", outputCol="song_index", handleInvalid="keep")

    # Fit and transform the data to convert IDs to numeric indices
    user_indexer_model = user_indexer.fit(triplets_df)
    song_indexer_model = song_indexer.fit(triplets_df)

    triplets_df = user_indexer_model.transform(triplets_df)
    triplets_df = song_indexer_model.transform(triplets_df)

    # Prepare the data for SVD model (using ALS for training)
    (training_df, test_df) = triplets_df.randomSplit([0.8, 0.2])

    # Initialize the ALS model for SVD (as a workaround)
    als = ALS(
        userCol="user_index",
        itemCol="song_index",
        ratingCol="listen_count",
        coldStartStrategy="drop",
        rank=10,
        maxIter=10,
        regParam=0.01,
        nonnegative=True  # Ensures non-negative factors for SVD-like behavior
    )

    # Fit the model
    model_start_time = time.time()
    model = als.fit(training_df)
    model_end_time = time.time()

    # Generate top N recommendations for all users
    user_recs = model.recommendForAllUsers(num_recommendations)

    # Evaluate the model using RMSE
    predictions = model.transform(test_df)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="listen_count", predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)
    print(f"Root Mean Square Error (RMSE): {rmse}")

    # Find the song_index for the given song_name
    song_index_df = song_data_df.filter(col("title").contains(song_name))

    if song_index_df.count() > 0:
        song_index = song_indexer_model.transform(song_index_df).select("song_index").first()[0]

        # Filter recommendations for the specific song_index
        user_recs_filtered = user_recs.filter(col("user_index") == song_index)

        # Extract recommendations
        recommendations = user_recs_filtered.select(explode("recommendations").alias("recommendation")).collect()

        if recommendations:
            # Convert to DataFrame
            recommendations_df = spark.createDataFrame(
                [(rec.recommendation.song_index, rec.recommendation.rating) for rec in recommendations],
                ["song_index", "rating"])

            # Join with the original song_data_df using a mapping DataFrame
            song_index_mapping = song_indexer_model.transform(song_data_df)
            result_df = recommendations_df.join(song_index_mapping, on="song_index").select("song_id", "title",
                                                                                            "artist_name", "rating")

            # Show the recommendations
            for rec in result_df.collect():
                print(
                    f"Recommended song ID: {rec.song_id}, Predicted listen count: {rec.rating:.2f}")
        else:
            print(f"No recommendations found for song name '{song_name}'")
    else:
        print(f"Song name '{song_name}' not found in the data.")

    end_time = time.time()
    total_time = end_time - start_time
    model_time = model_end_time - model_start_time

    print(f"Total time taken: {total_time:.2f} seconds")
    print(f"Time taken to fit the model: {model_time:.2f} seconds")

    spark.stop()

if __name__ == "__main__":
    song_name = "Despacito"  # Replace with the song name for which you want recommendations
    song_data_path = "song_data_json"  # Path to the JSON directory for song data
    user_data_path = "user_data_json"  # Path to the JSON directory for user data

    recommend_music(song_name, song_data_path, user_data_path, num_recommendations=1)
