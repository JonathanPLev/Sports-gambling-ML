import requests
from datetime import datetime
import pandas as pd

try:
    # Make a GET request to the API endpoint
    headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.0.0 Safari/537.36"

    }
    session = requests.Session()
    response = session.get('https://api.prizepicks.com/projections', headers=headers)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        try:
            # Parse the JSON data from the response
            data = response.json()

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
                player_name = projection["relationships"]["new_player"]["data"]["id"]
                line_score = projection["attributes"]["line_score"]
                game_id = projection["attributes"]["game_id"]
                projection_data = {
                    "Player Name": player_name,
                    "Line Score": line_score,
                    "Game ID": game_id
                }
                projections_data.append(projection_data)

            # Create a DataFrame from the projections data
            df = pd.DataFrame(projections_data)

            # Define the filename with the current date and time
            filename = f"projections_{current_date}_{current_time}.csv"

            # Save the DataFrame to a CSV file
            df.to_csv(filename, index=False)

            print(f"Data saved to {filename}")
        except requests.exceptions.JSONDecodeError as e:
            print("Error decoding JSON response:", str(e))
    else:
        print(f"Request failed with status code {response.status_code}")
except requests.exceptions.RequestException as e:
    print("Error making the request:", str(e))