import pandas as pd
import openai

openai.api_key = "YOUR_API_KEY"
#path = "copy the path of the file from your drive after mounting with the drive"

# Load data from CSV file
df = pd.read_csv("your_data_file.csv")
#or df = pd.read_csv(path)

# Initialize an empty list to store the generated text
generated_prompt = []
generated_completion = []
# Loop through each row in the dataset


for index, row in df.iterrows():
    # Create a prompt with the movie name
    prompt = f"Please write a short summary for the movie {row['movie_name']}."
    
    # Create a completion with the movie details
    completion = f"{row['movie_details']}. This movie was released on {row['release_date']} and directed by {row['movie_director']}. Its stars are {row['actors']} and belongs to the {row['movie_genre']} genre."
    
    # Append the prompt and completion to the generated text list
    generated_prompt.append(prompt )
    generated_completion.append(completion)


# Add the generated text to a new column in the dataframe
df['generated_prompt'] = generated_prompt
df['generated_completion'] = generated_completion


# Save the modified dataframe to a new CSV file
df.to_csv("prepareddata.csv", index=False)

