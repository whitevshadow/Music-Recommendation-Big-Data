from pyspark.sql import SparkSession
from pyspark.sql.functions import col, struct, collect_list
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
import os

# Set Spark Local IP if needed
os.environ['SPARK_LOCAL_IP'] = '192.168.0.103'

def create_json_files(song_data_path, triplets_file_path, song_output_json_path, user_output_json_path):
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

    # Define the schema for the song_data.csv file
    song_data_schema = StructType([
        StructField("song_id", StringType(), True),
        StructField("title", StringType(), True),
        StructField("release", StringType(), True),
        StructField("artist_name", StringType(), True),
        StructField("year", IntegerType(), True)
    ])

    # Define the schema for the triplets_file.csv file
    triplets_schema = StructType([
        StructField("user_id", StringType(), True),
        StructField("song_id", StringType(), True),
        StructField("listen_count", IntegerType(), True)
    ])

    # Load the song_data.csv file
    song_data_df = spark.read.csv(song_data_path, schema=song_data_schema, header=True)

    # Load the triplets_file.csv file
    triplets_df = spark.read.csv(triplets_file_path, schema=triplets_schema, header=True)

    # Perform data cleaning
    song_data_df = song_data_df.dropDuplicates()
    triplets_df = triplets_df.dropDuplicates()

    # Handle missing values
    song_data_df = song_data_df.fillna({'year': 0})
    triplets_df = triplets_df.fillna({'listen_count': 0})

    # Create JSON file for song data
    song_data_df.write.mode("overwrite").json(song_output_json_path)
    print(f"Song JSON created at: {song_output_json_path}")

    # Join the triplets_df with the song_data_df to get song details
    enriched_triplets_df = triplets_df.join(song_data_df, "song_id", "left")

    # Group by user_id and collect songs into an array
    user_songs_df = enriched_triplets_df.groupBy("user_id").agg(
        collect_list(struct("song_id", "title", "artist_name", "year")).alias("songs")
    ).repartition(10)  # Adjust the number of partitions

    # Convert the DataFrame to JSON format
    user_songs_df.write.mode("overwrite").json(user_output_json_path)
    print(f"User JSON created at: {user_output_json_path}")

    spark.stop()

if __name__ == "__main__":
    song_data_path = "song_data.csv"  # Path to the song_data.csv file
    triplets_file_path = "triplets_file.csv"  # Path to the triplets_file.csv file
    song_output_json_path = "song_data_json"  # Output path for the song JSON files
    user_output_json_path = "user_data_json"  # Output path for the user JSON files

    create_json_files(song_data_path, triplets_file_path, song_output_json_path, user_output_json_path)
