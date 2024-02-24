#!/usr/bin/env python
# coding: utf-8

# In[10]:


import requests
from bs4 import BeautifulSoup
import random

def scrapeWikiArticle(url):
   response = requests.get(
     url=url,
    )

   soup = BeautifulSoup(response.content, 'html.parser')

   title = soup.find(id="firstHeading")
   print(title.text)

   allLinks = soup.find(id="bodyContent").find_all("a")
   random.shuffle(allLinks)
   linkToScrape = 0

   for link in allLinks:
   # We are only interested in other wiki articles
    if link['href'].find("/wiki/") == -1: 
     continue

   # Use this link to scrape
    linkToScrape = link
    break

    scrapeWikiArticle("https://en.wikipedia.org" + linkToScrape['href'])

scrapeWikiArticle(" https://en.wikipedia.org/wiki/List_of_most-viewed_YouTube_videos")


# In[13]:


import requests
from bs4 import BeautifulSoup
url = "https://www.bcci.tv/"
response = requests.get(url)
soup = BeautifulSoup(response.content, "html.parser")
fixtures_link = soup.find("a", text="International Fixtures")[href]
fixtures_url = url + fixtures_link
fixtures_response = requests.get(fixtures_url)
fixtures_soup = BeautifulSoup(fixtures_response.content, "html.parser")
fixtures = fixtures_soup.find_all("div", class_="fixture__format-strip")

for fixture in fixtures:
  series = fixture.find("span", class_="u-unskewed-text").text.strip()
  place = fixture.find("p", class_="fixture__additional-info").text.strip()
  date = fixture.find("span", class_="fixture__date").text.strip()
  time = fixture.find("span", class_="fixture__time").text.strip()

  print("Series:", series)
  print("Place:", place)
  print("Date:", date)
  print("Time:", time)
  print()


# In[16]:


import requests
from bs4 import BeautifulSoup

url = "https://www.imdb.com/search/title/?title_type=tv_series&sort=num_votes,desc"

# Send a GET request to the URL
response = requests.get(url)

# Create a BeautifulSoup object
soup = BeautifulSoup(response.text, 'html.parser')

# Find the container that holds the trending repositories
trending_repos = soup.find_all('article', class_='Box-row')

# Iterate over each repository
for repo in trending_repos:
  # Find the repository title
  title = repo.find('h1', class_='h3').text.strip()

  # Find the repository description
  description = repo.find('p', class_='col-9').text.strip()

  # Find the contributors count
  contributors = repo.find('a', class_='muted-link').text.strip()

  # Find the language used
  language = repo.find('span', itemprop='programmingLanguage').text.strip()

  # Print the details
  print("Repository Title:", title)
  print("Repository Description:", description)
  print("Contributors Count:", contributors)
  print("Language Used:", language)
  print()


# In[17]:


import requests
from bs4 import BeautifulSoup

r = requests.get('https://www.billboard.com/charts/hot-100/')
soup = BeautifulSoup(r.content, 'html.parser')
result = soup.find_all('div', class_='o-chart-results-list-row-container')
for res in result:
    songName = res.find('h3').text.strip()
    artist = res.find('h3').find_next('span').text.strip()
    print("song: "+songName)
    print("artist: "+ str(artist))
    print("___________________________________________________")


# In[18]:


import requests
from bs4 import BeautifulSoup

# Send a GET request to the URL
url = "https://www.theguardian.com/news/datablog/2012/aug/09/best-selling-books-all-time-fifty-shades-grey-compare"
response = requests.get(url)

# Parse the HTML content
soup = BeautifulSoup(response.content, 'html.parser')

# Find the relevant HTML elements and extract the data
novels = []
table = soup.find('table')
rows = table.find_all('tr')[1:]  # Exclude the header row

for row in rows:
  columns = row.find_all('td')
  book_name = columns[1].text.strip()
  author_name = columns[2].text.strip()
  volumes_sold = columns[3].text.strip()
  publisher = columns[4].text.strip()
  genre = columns[5].text.strip()

  novel = {
  'Book Name': book_name,
  'Author Name': author_name,
  'Volumes Sold': volumes_sold,
  'Publisher': publisher,
  'Genre': genre
  }
  novels.append(novel)

# Print the scraped data
for novel in novels:
  print(novel)


# In[19]:


import requests
from bs4 import BeautifulSoup

url = "https://www.imdb.com/list/ls095964455/"
response = requests.get(url)
soup = BeautifulSoup(response.content, "html.parser")

series_list = soup.find_all("div", class_="lister-item-content")

for series in series_list:
  name = series.find("h3").find("a").text.strip()
  year_span = series.find("span", class_="lister-item-year").text.strip("()")

  genre = series.find("span", class_="genre").text.strip()
  runtime = series.find("span", class_="runtime").text.strip()
  rating = series.find("span", class_="ipl-rating-star__rating").text.strip()
  votes = series.find("span", attrs={"name": "nv"}).text.strip()

  print("Name:", name)
  print("Year Span:", year_span)
  print("Genre:", genre)
  print("Run Time:", runtime)
  print("Ratings:", rating)
  print("Votes:", votes)
  print()


# In[22]:


import lxml
import re
import numpy as np
import pandas as pd

from bs4 import BeautifulSoup
from requests import get
url1 = "https://www.imdb.com/search/title?count=100&title_type=feature,tv_series&ref_=nv_wl_img_2"
class IMDB(object):
	"""docstring for IMDB"""
	def __init__(self, url):
		super(IMDB, self).__init__()
		page = get(url)

		self.soup = BeautifulSoup(page.content, 'lxml')

	def articleTitle(self):
		return self.soup.find("h1", class_="header").text.replace("\n","")

	def bodyContent(self):
		content = self.soup.find(id="main")
		return content.find_all("div", class_="lister-item mode-advanced")

	def movieData(self):
		movieFrame = self.bodyContent()
		movieTitle = []
		movieDate = []
		movieRunTime = []
		movieGenre = []
		movieRating = []
		movieScore = []
		movieDescription = []
		movieDirector = []
		movieStars = []
		movieVotes = []
		movieGross = []
		for movie in movieFrame:
			movieFirstLine = movie.find("h3", class_="lister-item-header")
			movieTitle.append(movieFirstLine.find("a").text)
			movieDate.append(re.sub(r"[()]","", movieFirstLine.find_all("span")[-1].text))
			try:
				movieRunTime.append(movie.find("span", class_="runtime").text[:-4])
			except:
				movieRunTime.append(np.nan)
			movieGenre.append(movie.find("span", class_="genre").text.rstrip().replace("\n","").split(","))
			try:
				movieRating.append(movie.find("strong").text)
			except:
				movieRating.append(np.nan)
			try:
				movieScore.append(movie.find("span", class_="metascore unfavorable").text.rstrip())
			except:
				movieScore.append(np.nan)
			movieDescription.append(movie.find_all("p", class_="text-muted")[-1].text.lstrip())
			movieCast = movie.find("p", class_="")

			try:
				casts = movieCast.text.replace("\n","").split('|')
				casts = [x.strip() for x in casts]
				casts = [casts[i].replace(j, "") for i,j in enumerate(["Director:", "Stars:"])]
				movieDirector.append(casts[0])
				movieStars.append([x.strip() for x in casts[1].split(",")])
			except:
				casts = movieCast.text.replace("\n","").strip()
				movieDirector.append(np.nan)
				movieStars.append([x.strip() for x in casts.split(",")])

			movieNumbers = movie.find_all("span", attrs={"name": "nv"})

			if len(movieNumbers) == 2:
				movieVotes.append(movieNumbers[0].text)
				movieGross.append(movieNumbers[1].text)
			elif len(movieNumbers) == 1:
				movieVotes.append(movieNumbers[0].text)
				movieGross.append(np.nan)
			else:
				movieVotes.append(np.nan)
				movieGross.append(np.nan)

		movieData = [movieTitle, movieDate, movieRunTime, movieGenre, movieRating, movieScore, movieDescription,
							movieDirector, movieStars, movieVotes, movieGross]
		return movieData
    
if __name__ == '__main__':
	site1 = IMDB(url1)
	print("Subject: ", site1.articleTitle())
	data = site1.movieData()
	for i in range(len(data)):
		print(data[i][:]) #Print the data


# In[23]:


import requests
from bs4 import BeautifulSoup

# Send a GET request to the UCI machine learning repositories website
url = "https://archive.ics.uci.edu/"
response = requests.get(url)

# Parse the HTML content using BeautifulSoup
soup = BeautifulSoup(response.content, "html.parser")

# Find the link to the "Show All Dataset" page
show_all_link = soup.find("a", href="ml/datasets.php")

# Construct the URL for the "Show All Dataset" page
show_all_url = url + show_all_link["href"]

# Send another GET request to the "Show All Dataset" page
show_all_response = requests.get(show_all_url)

# Parse the HTML content of the "Show All Dataset" page
show_all_soup = BeautifulSoup(show_all_response.content, "html.parser")

# Find the table containing the dataset details
dataset_table = show_all_soup.find("table", class_="table")

# Extract the details from the table rows
dataset_details = []
for row in dataset_table.find_all("tr")[1:]:
  columns = row.find_all("td")
  dataset_name = columns[0].text.strip()
  data_type = columns[1].text.strip()
  task = columns[2].text.strip()
  attribute_type = columns[3].text.strip()
  num_instances = columns[4].text.strip()
  num_attributes = columns[5].text.strip()
  year = columns[6].text.strip()
  dataset_details.append((dataset_name, data_type, task, attribute_type, num_instances, num_attributes, year))

# Print the dataset details
for dataset in dataset_details:
  print("Dataset Name:", dataset[0])
  print("Data Type:", dataset[1])
  print("Task:", dataset[2])
  print("Attribute Type:", dataset[3])
  print("No of Instances:", dataset[4])
  print("No of Attributes:", dataset[5])
  print("Year:", dataset[6])
  print()


# In[ ]:




