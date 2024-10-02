import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
import pandas as pd
from datetime import datetime

# Path to ChromeDriver installed via Homebrew
driver_path = '/opt/homebrew/bin/chromedriver'

# Set up Chrome options
options = webdriver.ChromeOptions()
options.add_argument('--headless')  # Run headless Chrome
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')

# Create a Service object
service = Service(executable_path=driver_path)

# Initialize the Chrome WebDriver with the Service and options
driver = webdriver.Chrome(service=service, options=options)

# Define the URL of the PrizePicks website
url = 'https://www.prizepicks.com/'

# Open the URL
driver.get(url)

# Wait for the pop-up close button and close it if it appears
try:
    WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, '//*[@id="close-popup-button"]'))).click()
except:
    print("No pop-up found or could not close pop-up.")

# Navigate to the NBA props section (adjust the selector as needed)
nba_props_button = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.LINK_TEXT, 'NBA')))
nba_props_button.click()

# Wait for the NBA props page to load
WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, '//*[@id="players-table"]')))

# Scrape the data
ppPlayers = []
players_table = driver.find_element(By.XPATH, '//*[@id="players-table"]')
players = players_table.find_elements(By.TAG_NAME, 'tr')

for player in players:
    player_data = player.find_elements(By.TAG_NAME, 'td')
    player_info = [data.text for data in player_data]
    ppPlayers.append(player_info)

# Close the WebDriver
driver.quit()

# Convert the data to a DataFrame
df = pd.DataFrame(ppPlayers, columns=['Player', 'Prop', 'Value'])

# Get the current date and time
current_time = datetime.now().strftime('%Y%m%d_%H%M%S')

# Create the filename with the current date and time
filename = f'prizepicks_data_{current_time}.csv'

# Save the DataFrame to a CSV file
df.to_csv(filename, index=False)

print(f"Data successfully saved to {filename}")
