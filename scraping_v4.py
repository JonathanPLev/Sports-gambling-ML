import json
import pandas as pd
from datetime import datetime

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

# Load the JSON data from a local file
with open('projections.json', 'r') as file:
    data = json.load(file)

# Get the current date and time
current_date = datetime.now().strftime("%Y-%m-%d")
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

# Filter the projections based on the current date and projection type
filtered_projections = []
for projection in data["data"]:
    updated_at = projection["attributes"]["updated_at"]
    projection_type = projection["attributes"]["projection_type"]
    if updated_at.startswith(current_date) and projection_type == "Single Stat":
        filtered_projections.append(projection)

# Extract the relevant information from the filtered projections
projections_data = []
for projection in filtered_projections:
    player_id = projection["relationships"]["new_player"]["data"]["id"]
    player_name = ""  # Initialize player_name as an empty string
    
    # Find the player name based on the player ID
    for included_data in data.get("included", []):
        if included_data["type"] == "new_player" and included_data["id"] == player_id:
            player_name = included_data["attributes"]["name"]
            break
    
    line_score = projection["attributes"]["line_score"]
    start_time = projection["attributes"]["start_time"]
    stat_type = projection["attributes"].get("stat_type", "")
    game_id = projection["attributes"].get("game_id", "")  # Use get() with a default value of empty string
    league_id = projection["relationships"]["league"]["data"]["id"]

    
    # Find the league/sport name based on the league ID
    league_name = ""
    for included_data in data.get("included", []):
        if included_data["type"] == "league" and included_data["id"] == league_id:
            league_name = included_data["attributes"]["name"]
            break
    
    # Exclude projections when league_name is "SPECIALS"
    if league_name != "SPECIALS":
        projection_data = {
            "Player ID": player_id,
            "Player Name": player_name,
            "Line Score": line_score,
            "Game ID": game_id,
            "Stat Type": stat_type,
            "League/Sport": league_name
        }
        projections_data.append(projection_data)

# Create a DataFrame from the projections data
df = pd.DataFrame(projections_data)

# Define the filename with the current date and time
filename = f"projections_{current_date}_{current_time}.csv"

# Save the DataFrame to a CSV file
df.to_csv(filename, index=False)

print(f"Data saved to {filename}")