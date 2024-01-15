#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
from bs4 import BeautifulSoup
import pandas as pd

# Send a GET request to the Wikipedia page
url = "https://en.wikipedia.org/wiki/Main_Page"
response = requests.get(url)

# Create a BeautifulSoup object to parse the HTML content
soup = BeautifulSoup(response.content, "html.parser")

# Find all the header tags (h1 to h6) using the find_all method
header_tags = soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])

# Extract the text from the header tags and store them in a list
header_texts = [tag.get_text() for tag in header_tags]

# Create a data frame using pandas
df = pd.DataFrame(header_texts, columns=["Header"])

# Display the data frame
print(df)


# In[2]:


import requests
from bs4 import BeautifulSoup
import pandas as pd

# Send a GET request to the website
url = "https://presidentofindia.nic.in/"
response = requests.get(url)
print (response)
# Create a BeautifulSoup object to parse the HTML content
soup = BeautifulSoup(response.content, "html.parser")
print(soup)
# Find the table containing the information
#table = soup.find("href")
print(soup.find("class"))
# Create empty lists to store the data
names = []
terms = []

# Iterate over each row in the table
for row in table.find_all("tr")[1:]:
  # Extract the name and term of office from the columns
  columns = row.find_all("td")
  name = columns[0].text.strip()
  term = columns[1].text.strip()
  
  # Append the data to the respective lists
  names.append(name)
  terms.append(term)

# Create a data frame using the lists
data = {"Name": names, "Term of Office": terms}
df = pd.DataFrame(data)

# Display the data frame
print(df)


# In[3]:


url = "https://www.icc-cricket.com/rankings/mens/player-rankings/odi/batting"
response = requests.get(url)
soup = BeautifulSoup(response.content, "html.parser")

batsman_data = []
table = soup.find("table", class_="table")
rows = table.find_all("tr")

for row in rows[1:11]:
  cells = row.find_all("td")
  batsman = cells[1].text.strip()
  team = cells[2].text.strip()
  rating = cells[3].text.strip()
  batsman_data.append([batsman, team, rating])

df = pd.DataFrame(batsman_data, columns=["Batsman", "Team", "Rating"])
print(df)


# In[4]:


url = "https://www.icc-cricket.com/rankings/mens/player-rankings/odi/bowling"
response = requests.get(url)
soup = BeautifulSoup(response.content, "html.parser")

bowler_data = []
table = soup.find("table", class_="table")
rows = table.find_all("tr")

for row in rows[1:11]:
  cells = row.find_all("td")
  bowler = cells[1].text.strip()
  team = cells[2].text.strip()
  rating = cells[3].text.strip()
  bowler_data.append([bowler, team, rating])

df = pd.DataFrame(bowler_data, columns=["Bowler", "Team", "Rating"])
print(df)


# In[5]:


import requests
from bs4 import BeautifulSoup
import pandas as pd

# Scrape Top 10 ODI teams in women's cricket
url_teams = "https://www.icc-cricket.com/rankings/womens/team-rankings/odi"
response_teams = requests.get(url_teams)
soup_teams = BeautifulSoup(response_teams.content, "html.parser")

teams_data = []
table_teams = soup_teams.find("table", class_="table")
rows_teams = table_teams.find_all("tr")

for row in rows_teams[1:11]:
  team_name = row.find("span", class_="u-hide-phablet").text.strip()
  matches = row.find_all("td")[2].text.strip()
  points = row.find_all("td")[3].text.strip()
  rating = row.find_all("td")[4].text.strip()
  teams_data.append([team_name, matches, points, rating])

# Scrape Top 10 women's ODI Batting players
url_batting = "https://www.icc-cricket.com/rankings/womens/player-rankings/odi/batting"
response_batting = requests.get(url_batting)
soup_batting = BeautifulSoup(response_batting.content, "html.parser")

batting_data = []
table_batting = soup_batting.find("table", class_="table")
rows_batting = table_batting.find_all("tr")

for row in rows_batting[1:11]:
  player_name = row.find("td", class_="table-body__cell rankings-table__name name").text.strip()
  team = row.find("span", class_="table-body__logo-text").text.strip()
  rating = row.find("td", class_="table-body__cell rating").text.strip()
  batting_data.append([player_name, team, rating])

# Scrape Top 10 women's ODI all-rounders
url_allrounders = "https://www.icc-cricket.com/rankings/womens/player-rankings/odi/all-rounder"
response_allrounders = requests.get(url_allrounders)
soup_allrounders = BeautifulSoup(response_allrounders.content, "html.parser")

allrounders_data = []
table_allrounders = soup_allrounders.find("table", class_="table")
rows_allrounders = table_allrounders.find_all("tr")

for row in rows_allrounders[1:11]:
  player_name = row.find("td", class_="table-body__cell rankings-table__name name").text.strip()
  team = row.find("span", class_="table-body__logo-text").text.strip()
  rating = row.find("td", class_="table-body__cell rating").text.strip()
  allrounders_data.append([player_name, team, rating])

# Create data frames
df_teams = pd.DataFrame(teams_data, columns=["Team", "Matches", "Points", "Rating"])
df_batting = pd.DataFrame(batting_data, columns=["Player", "Team", "Rating"])
df_allrounders = pd.DataFrame(allrounders_data, columns=["Player", "Team", "Rating"])

# Print the data frames
print("Top 10 ODI teams in women's cricket:")
print(df_teams)
print("\nTop 10 women's ODI Batting players:")
print(df_batting)
print("\nTop 10 women's ODI all-rounders:")
print(df_allrounders)


# In[6]:


import requests
from bs4 import BeautifulSoup
import pandas as pd

# Send a GET request to the website
url = "https://www.cnbc.com/world/?region=world"
response = requests.get(url)

# Create a BeautifulSoup object to parse the HTML content
soup = BeautifulSoup(response.content, "html.parser")

# Find all the news articles on the page
articles = soup.find_all("div", class_="Card-titleContainer")

# Initialize empty lists to store the scraped data
headlines = []
times = []
links = []

# Loop through each article and extract the required information
for article in articles:
  # Extract the headline
  headline = article.find("a").text.strip()
  headlines.append(headline)
  
  # Extract the time
  time = article.find("time").text.strip()
  times.append(time)
  
  # Extract the news link
  link = article.find("a")["href"]
  links.append(link)

# Create a dataframe using the scraped data
data = {
  "Headline": headlines,
  "Time": times,
  "News Link": links
}
df = pd.DataFrame(data)

# Print the dataframe
print(df)


# In[7]:


import requests
from bs4 import BeautifulSoup
import pandas as pd

# Send a GET request to the URL
url = "https://www.journals.elsevier.com/artificial-intelligence/most-downloaded-articles"
response = requests.get(url)

# Create a BeautifulSoup object to parse the HTML content
soup = BeautifulSoup(response.content, "html.parser")

# Find the container that holds the article details
articles_container = soup.find("div", class_="pod-listing")

# Initialize empty lists to store the scraped data
titles = []
authors = []
dates = []
urls = []

# Iterate over each article in the container
for article in articles_container.find_all("li"):
  # Scrape the title
  title = article.find("h3").text.strip()
  titles.append(title)
  
  # Scrape the authors
  author = article.find("span", class_="text-xs").text.strip()
  authors.append(author)
  
  # Scrape the published date
  date = article.find("span", class_="text-xs").find_next_sibling("span").text.strip()
  dates.append(date)
  
  # Scrape the paper URL
  url = article.find("a")["href"]
  urls.append(url)

# Create a dataframe with the scraped data
data = {
  "Paper Title": titles,
  "Authors": authors,
  "Published Date": dates,
  "Paper URL": urls
}
df = pd.DataFrame(data)

# Print the dataframe
print(df)


# In[8]:


import requests
from bs4 import BeautifulSoup
import pandas as pd

# Send a GET request to the website
url = "https://www.dineout.co.in"
response = requests.get(url)

# Create a BeautifulSoup object to parse the HTML content
soup = BeautifulSoup(response.content, 'html.parser')

# Find the elements containing the details you want to scrape
restaurant_names = soup.find_all('h2', class_='restnt-name ellipsis')
cuisines = soup.find_all('span', class_='double-line-ellipsis')
locations = soup.find_all('span', class_='double-line-ellipsis')
ratings = soup.find_all('span', class_='rating-value')
image_urls = soup.find_all('img', class_='img-responsive')

# Create empty lists to store the scraped data
restaurant_list = []
cuisine_list = []
location_list = []
rating_list = []
image_url_list = []

# Extract the data from the elements and append them to the respective lists
for name in restaurant_names:
  restaurant_list.append(name.text.strip())

for cuisine in cuisines:
  cuisine_list.append(cuisine.text.strip())

for location in locations:
  location_list.append(location.text.strip())

for rating in ratings:
  rating_list.append(rating.text.strip())

for image in image_urls:
  image_url_list.append(image['src'])

# Create a dictionary from the lists
data = {
  'Restaurant Name': restaurant_list,
  'Cuisine': cuisine_list,
  'Location': location_list,
  'Ratings': rating_list,
  'Image URL': image_url_list
}

# Create a dataframe from the dictionary
df = pd.DataFrame(data)

# Print the dataframe
print(df)


# In[ ]:




