**Music Recommendation System**

**Overview**
In recent years, music streaming services have revolutionized how users discover and consume music. With an expanding library of songs and diverse user preferences, effective music recommendation systems have become critical for enhancing user engagement and satisfaction. This project presents a music recommendation system utilizing Apache Spark for efficient data processing and Streamlit for an interactive user interface.
System Architecture
The architecture of the music recommendation system is designed to efficiently process large datasets and deliver personalized music suggestions. At its core, the system employs Apache Spark for data processing, which handles data ingestion from two primary sources:
A CSV file containing song metadata.
A CSV file detailing user-song interactions.
Data Processing
Data Ingestion: Load song and user data from CSV files.
Data Cleaning: Remove duplicates and handle missing values.
Data Transformation: Convert cleaned data into JSON format for efficient processing.
Data Enrichment: Join user interactions with song metadata to create a comprehensive dataset linking user preferences to specific songs.

**Recommendation Engine**

The core of the recommendation engine is built upon the Alternating Least Squares (ALS) algorithm, a popular method for collaborative filtering. This approach identifies patterns in user preferences, suggesting songs that users are likely to enjoy based on their past interactions.
User Interface
Developed using Streamlit, the user interface allows users to input a song name and receive personalized recommendations. The interface displays:
Recommended songs.
Album covers sourced from the Spotify API, enhancing the visual context of the recommendations.
Data Requirements
The system utilizes two main datasets:
song_data.csv: Contains metadata for songs (e.g., song_id, title, release year, artist_name).
user_data.csv: Captures user interactions with songs (e.g., user_id, song_id, listen_count).
After processing, these datasets are converted into JSON format for efficient querying and retrieval during recommendation generation.

![image](https://github.com/user-attachments/assets/ee409a48-e04a-42d2-84d8-cda8701b730a)


**Recommendation Techniques**

The system employs both collaborative and content-based filtering techniques:
Collaborative Filtering: Builds a model from users' past behavior and similar decisions made by other users.
Content-Based Filtering: Utilizes discrete characteristics of items to recommend additional items with similar properties.
Hybrid Filtering: Combines both techniques to enhance recommendation accuracy.
Challenges Addressed
The project also addresses various challenges in processing user-generated content:
Handling regional language data and SMS-style language.
Classifying reviews as positive or negative using SentiWordNet.
Identifying irrelevant or spam reviews.

![Screenshot 2024-08-30 124556](https://github.com/user-attachments/assets/e2fc37fe-bc3b-4793-b2df-d923a0b2f888)




**Conclusion**

This music recommendation system bridges the gap in existing models by incorporating a broader range of user data and contextual factors, ultimately leading to more personalized and relevant music recommendations. By leveraging big data technologies and an intuitive interface, the system enhances the overall user experience in music discovery.
