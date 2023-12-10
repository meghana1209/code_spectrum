import surprise as surprrise
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split
from surprise import accuracy

# Load the MovieLens 100k dataset
data = Dataset.load_builtin('ml-100k')

# Load data (you can replace this with your dataset)
data = Dataset.load_builtin('/content/ml-100k.zip')

# Define a reader object to parse the file
reader = Reader(line_format='user item rating timestamp', sep='\t')

# Load the data from the file using the reader
data = Dataset.load_from_file('/content/ml-100k.zip', reader=reader)

# Split the dataset into training and testing sets
trainset, testset = train_test_split(data, test_size=0.25)

# Use the KNNBasic collaborative filtering algorithm
sim_options = {
    'name': 'cosine',
    'user_based': True  # User-User collaborative filtering
}

algo = KNNBasic(sim_options=sim_options)

# Train the algorithm on the training set
algo.fit(trainset)

# Make predictions on the testing set
predictions = algo.test(testset)

# Evaluate the model's performance
accuracy.rmse(predictions)

# Function to get movie recommendations for a user
def get_movie_recommendations(user_id, n=5):
    # Get a list of tuples (movie_id, predicted_rating)
    movie_predictions = [(iid, algo.predict(user_id, iid).est) for iid in trainset.all_items()]

    # Sort the list by predicted rating in descending order
    movie_predictions.sort(key=lambda x: x[1], reverse=True)

    # Return the top n movie recommendations
    top_recommendations = movie_predictions[:n]
    return top_recommendations

# Example: Get movie recommendations for user 1
user_id = '1'
recommendations = get_movie_recommendations(user_id, n=5)
print(f"Top 5 movie recommendations for user {user_id}:")
for movie_id, predicted_rating in recommendations:
    print(f"Movie ID: {movie_id}, Predicted Rating: {predicted_rating}")


# Create a train-test split
trainset, testset = train_test_split(data, test_size=0.25)

# Use the KNNBasic collaborative filtering algorithm
sim_options = {
    'name': 'cosine',
    'user_based': True
}

algo = KNNBasic(sim_options=sim_options)

# Train the algorithm on the training set
algo.fit(trainset)

# Make predictions on the testing set
predictions = algo.test(testset)

# Evaluate the model's performance
accuracy.rmse(predictions)
