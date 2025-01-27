import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, explode, when
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.ml.feature import StringIndexer

class MusicDataset(Dataset):
    def __init__(self, user_ids, song_ids, ratings):
        self.user_ids = user_ids
        self.song_ids = song_ids
        self.ratings = ratings

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        return {
            'user_id': self.user_ids[idx],
            'song_id': self.song_ids[idx],
            'rating': self.ratings[idx]
        }

class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_songs, embedding_dim):
        super(MatrixFactorization, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.song_embedding = nn.Embedding(num_songs, embedding_dim)

    def forward(self, user_ids, song_ids):
        user_factors = self.user_embedding(user_ids)
        song_factors = self.song_embedding(song_ids)
        return (user_factors * song_factors).sum(dim=1)

def train_model(song_data_path, user_data_path, model_path):
    # Create a Spark session with optimized configurations
    spark = SparkSession.builder \
        .appName("Music Recommendation System") \
        .config("spark.executor.memory", "8g") \
        .config("spark.executor.cores", "4") \
        .config("spark.driver.memory", "8g") \
        .config("spark.driver.cores", "4") \
        .config("spark.ui.port", "4057") \
        .config("spark.ui.enabled", "false") \
        .config("spark.driver.bindAddress", "127.0.0.1") \
        .config("spark.driver.host", "127.0.0.1") \
        .getOrCreate()

    # Set logging level to suppress warnings
    spark.sparkContext.setLogLevel("ERROR")

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

    # Get the maximum user and song indices
    max_user_id = triplets_df.select("user_index").rdd.max()[0]
    max_song_id = triplets_df.select("song_index").rdd.max()[0]

    # Prepare the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = MatrixFactorization(max_user_id + 1, max_song_id + 1, embedding_dim=5)
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Prepare data for PyTorch in chunks
    batch_size = 1000
    num_batches = triplets_df.count() // batch_size + 1

    for batch in range(num_batches):
        start = batch * batch_size
        end = min((batch + 1) * batch_size, triplets_df.count())
        batch_df = triplets_df.limit(end).offset(start)

        user_ids = batch_df.select("user_index").rdd.flatMap(lambda x: x).collect()
        song_ids = batch_df.select("song_index").rdd.flatMap(lambda x: x).collect()
        ratings = batch_df.select("listen_count").rdd.flatMap(lambda x: x).collect()

        if not user_ids or not song_ids or not ratings:
            continue  # Skip empty batches

        dataset = MusicDataset(user_ids, song_ids, ratings)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)  # Use multiple workers

        # Train the model
        for epoch in range(5):  # Reduced number of epochs
            for batch in dataloader:
                user_ids = batch['user_id'].to(device)
                song_ids = batch['song_id'].to(device)
                ratings = batch['rating'].to(device)
                optimizer.zero_grad()
                predictions = model(user_ids, song_ids)
                loss = criterion(predictions, ratings)
                loss.backward()
                optimizer.step()
                print(f'Epoch {epoch+1}, Batch {batch}, Loss: {loss.item()}')

    # Save the model
    torch.save(model.state_dict(), model_path)

    spark.stop()

# Example usage
song_data_path = "song_data_json"  # Path to the JSON directory for song data
user_data_path = "user_data_json"  # Path to the JSON directory for user data
model_path = "music_recommendation_model.pth"
train_model(song_data_path, user_data_path, model_path)
