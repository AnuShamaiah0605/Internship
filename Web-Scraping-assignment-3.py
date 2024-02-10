#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import selenium
from bs4 import BeautifulSoup
import time
from selenium import webdriver
import requests
import re
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException
from selenium.webdriver.support.ui import WebDriverWait


# In[6]:


# connecting to the webdriver
driver=webdriver.Chrome(r"C:\Users\Anu Shamaiah Prasad\Downloads\chromedriver_win32\chromedriver.exe")


# In[8]:


# getting the webpage of mentioned url
url = "https://www.amazon.in/"
driver.get(url)
user_input = input('Enter the product that we want to search : ')


# In[9]:


search = driver.find_element_by_id("twotabsearchtextbox")
search

# sending the user input to search bar
search.send_keys(user_input)

# locating the search button using xpath
search_btn = driver.find_element_by_xpath("//div[@class='nav-search-submit nav-sprite']/span/input")

# clicking on search button
search_btn.click()


# In[10]:


urls = []          # empty list
for i in range(0,3):      # for loop to scrape 3 pages
    page_url = driver.find_elements_by_xpath("//a[@class='a-link-normal a-text-normal']")
    for i in page_url:
        urls.append(i.get_attribute("href"))
        next_btn = driver.find_element_by_xpath("//li[@class='a-last']/a")
        time.sleep(3)


# In[11]:


len(urls)


# In[12]:


# making empty list and fetching required data
brand_name = []
product_name = []
ratings = []
num_ratings = []
prices = []
exchange = []
exp_delivery = []
availability = []
other_details = []

for i in urls:
    driver.get(i)
    time.sleep(3)
    
    
    #fetching brand name 
    try:
        brand = driver.find_element_by_xpath("//a[@id='bylineInfo']")
        brand_name.append(brand.text)
    except NoSuchElementException:
        brand_name.append('-')
    
    
    # fetching Name of the Product
    try:
        product = driver.find_element_by_xpath("//span[@id='productTitle']")
        product_name.append(product.text)
    except NoSuchElementException:
        product_name.append('-')
        
        

     #fetching ratings
    try:
        rating = driver.find_element_by_xpath("//span[@class='a-size-base a-nowrap']/span")
        ratings.append(rating.text)
    except NoSuchElementException:
        ratings.append('-')
        
 
    #fetching  no of ratings
    try:
        num_rating = driver.find_element_by_xpath("//span[@id='acrCustomerReviewText']")
        num_ratings.append(num_rating.text)
    except NoSuchElementException:
        num_ratings.append('-')
        

    #fetching price of the product
    try:
        price = driver.find_element_by_xpath("//td[@class='a-span12']")
        prices.append(price.text)
    except NoSuchElementException:
        prices.append('-')
        
        
    #fetching return/exchange
    try:
        exch = driver.find_element_by_xpath("//span[@class='a-declarative']/div/a")
        exchange.append(exch.text)
    except NoSuchElementException:
        exchange.append('-')
        

    #fetching expected delivery
    try:
        delivery = driver.find_element_by_xpath("//div[@class='a-section a-spacing-mini']/b")
        exp_delivery.append(delivery.text)
    except NoSuchElementException:
        exp_delivery.append('-')
        

    #fetching availability information
    try:
        avail = driver.find_element_by_xpath("//span[@class='a-size-medium a-color-success']")
        availability.append(avail.text)
    except NoSuchElementException:
        availability.append('-')
        
    #other details
    try:
        oth_det = driver.find_element_by_xpath("//ul[@class='a-unordered-list a-vertical a-spacing-mini']")
        other_details.append(oth_det.text)
    except NoSuchElementException:
        other_details.append('-')


# In[13]:


print(len(brand_name),
len(product_name),
len(ratings),
len(num_ratings),
len(prices),
len(exchange),
len(exp_delivery),
len(availability),
len(other_details))


# In[14]:


# Creating the DataFrame for the scraped data

guitar = pd.DataFrame({})
guitar['Brand Name'] = brand_name
guitar['Name of the Product'] = product_name
guitar['Rating'] = ratings
guitar['No. of Ratings'] = num_ratings
guitar['Price'] = prices
guitar['Return/Exchange'] = exchange
guitar['Expected Delivery'] = exp_delivery
guitar['Availability'] = availability
guitar['Other Details'] = other_details
guitar['Product URL'] = urls
guitar


# In[15]:


#saving the data in csv
guitar.to_csv("Guitar.csv")


# In[16]:


driver.close()


# In[17]:


# geting the webpage of mentioned url
url = "http://images.google.com/"

# creating empty list
urls = []
data = []

search_item = ["Fruits","Cars","Machine Learning"]
for item in search_item:
    driver.get(url)
    time.sleep(5)
    
    # finding webelement for search_bar
    search_bar = driver.find_element_by_tag_name("input")
    
    # sending keys to get the keyword for search bar
    search_bar.send_keys(str(item))
    
    # clicking on search button
    search_button = driver.find_element_by_xpath("//button[@class='Tg7LZd']").click()
    
    # scroling down the webpage to get some more images
    for _ in range(500):
        driver.execute_script("window.scrollBy(0,100)")
        
        imgs = driver.find_elements_by_xpath("//img[@class='rg_i Q4LuWd']")
    img_url = []
    for image in imgs:
        source = image.get_attribute('src')
        if source is not None:
            if(source[0:4] == 'http'):
                img_url.append(source)
    for i in img_url[:100]:
        urls.append(i)
        
for i in range(len(urls)):
    if i >= 300:
        break
    print("Doenloading {0} of {1} images" .format(i,300))
    response = requests.get(urls[i])
    
    file = open(r"E:\google\images"+str(i)+".jpg","wb")
    
    file.write(response.content)


# In[18]:


driver.close()


# In[19]:


# getting the webpage of mentioned url
url = "https://www.flipkart.com/"
driver.get(url)


# In[20]:


# closing login popup button
lonin_x_btn = driver.find_element_by_xpath("//div[@class='_2QfC02']//button").click()


# In[21]:


# search for web element
search_bar = driver.find_element_by_xpath("//input[@class='_3704LK']")

# sending keys to search product
search_bar.send_keys("pixel 4A")


# In[22]:


# location the search button using xpath
search_btn = driver.find_element_by_xpath("//button[@class='L0Z3Pu']")

# clicking on search button
search_btn.click()


# In[23]:


page1_url = []
urls = driver.find_elements_by_xpath("//a[@class='_1fQZEK']")
for url in urls:
    page1_url.append(url.get_attribute('href'))


# In[24]:


len(page1_url)


# In[25]:


# creating empty list
Smartphones = ({})
Smartphones['Brand'] = []
Smartphones['Phone name'] = []
Smartphones['Colour'] = []
Smartphones['RAM'] = []
Smartphones['Storage(ROM)'] = []
Smartphones['Primary Camera'] = []
Smartphones['Secondary Camera'] = []
Smartphones['Display Size'] = []
Smartphones['Display Resolution'] = []
Smartphones['Processor'] = []
Smartphones['Processor Cores'] = []
Smartphones['Battery Capacity'] = []
Smartphones['Price'] = []
Smartphones['URL'] = []


# In[26]:


# scraping data from each url of page 1
for url in page1_url:
    driver.get(url)
    print("Scraping URL = ",url)
    Smartphones['URL'].append(url)
    time.sleep(2)
    
    
    #clicking on read more button to get more information
    try:
        read_more = driver.find_element_by_xpath("//button[@class='_2KpZ6l _1FH0tX']")
        read_more.click()
    except NoSuchElementException:
        print("Exception occured while moving to next page")
    
    #scraping brand name of smartphone
    try:
        brand_tags = driver.find_element_by_xpath("//span[@class='B_NuCI']")
        Smartphones['Brand'].append(brand_tags.text.split()[0])
    except NoSuchElementException:
        Smartphones['Brand'].append('-')
    
    
    # scraping name of smartphones
    try:
        name_tags = driver.find_element_by_xpath("//div[@class='_3k-BhJ'][1]/table/tbody/tr[3]/td[2]/ul/li")
        Smartphones['Phone name'].append(name_tags.text)
    except NoSuchElementException:
        Smartphones['Phone name'].append('-')
        
    #scraping colour of smartphone
    try:
        color_tags = driver.find_element_by_xpath("//div[@class='_3k-BhJ'][1]/table/tbody/tr[4]/td[2]/ul/li")
        Smartphones['Colour'].append(color_tags.text)
    except NoSuchElementException:
        Smartphones['Colour'].append('-')
        
    # scraping RAM data of smartphone
    try:
        ram_tags = driver.find_element_by_xpath("//div[@class='_3k-BhJ'][4]/table[1]/tbody/tr[2]/td[2]/ul/li")
        Smartphones['RAM'].append(ram_tags.text)
    except NoSuchElementException:
        Smartphones['RAM'].append('-')
        
    #scraping ROM data of smartphones
    try:
        rom = driver.find_element_by_xpath("//div[@class='_3k-BhJ'][4]/table[1]/tbody/tr[1]/td[2]/ul/li")
        Smartphones['Storage(ROM)'].append(rom.text)
    except NoSuchElementException:
        Smartphones['Storage(ROM)'].append('-')
        
    # scraping  Primary camera data of smartphone
    try:
        pri =driver.find_element_by_xpath("//div[@class='_3k-BhJ'][5]/table[1]/tbody/tr[2]/td[2]/ul/li")
        Smartphones['Primary Camera'].append(pri.text)
    except NoSuchElementException:
        Smartphones['Primary Camera'].append('-')
        
    # scraping secondary camera data of smartphone
    try:
        sec = driver.find_element_by_xpath("//div[@class='_3k-BhJ'][5]/table[1]/tbody/tr[6]/td[1]")
        if sec != 'Secondary Camera' :
            if driver.find_element_by_xpath("//div[@class='_3k-BhJ'][5]/table[1]/tbody/tr[5]/td[1]").text == "Secondary Camera":
                sec_cam =driver.find_element_by_xpath("//div[@class='_3k-BhJ'][5]/table[1]/tbody/tr[5]/td[2]/ul/li")
            else :
                raise NoSuchElementException
        else :
            sec_cam = driver.find_element_by_xpath("//div[@class='_3k-BhJ'][5]/table[1]/tbody/tr[6]/td[2]/ul/li")
        Smartphones['Secondary Camera'].append(sec_cam.text)
    except NoSuchElementException:
        Smartphones['Secondary Camera'].append('-')
        
    
    #scraping display size data of smartphone
    try:
        disp = driver.find_element_by_xpath("//div[@class='_3k-BhJ'][2]/div")
        if disp.text != 'Display Features' : raise NoSuchElementException
        disp_size = driver.find_element_by_xpath("//div[@class='_3k-BhJ'][2]/table[1]/tbody/tr[1]/td[2]/ul/li")
        Smartphones['Display Size'].append(disp_size.text)
    except NoSuchElementException:
        Smartphones['Display Size'].append('-')
        
    
    #scraping display resolution of smartphone
    try:
        disp = driver.find_element_by_xpath("//div[@class='_3k-BhJ'][2]/div")
        if disp.text != 'Display Features' : raise NoSuchElementException
        disp_reso = driver.find_element_by_xpath("//div[@class='_3k-BhJ'][2]/table[1]/tbody/tr[2]/td[2]/ul/li")
        Smartphones['Display Resolution'].append(disp_reso.text)
    except NoSuchElementException:
        Smartphones['Display Resolution'].append('-')
        
        
    #scraping processor of smartphone
    try:
        pro = driver.find_element_by_xpath("//div[@class='_3k-BhJ'][3]/table[1]/tbody/tr[2]/td[1]]")
        if pro.text != 'Processor Type' : raise NoSuchElementException
        processor = driver.find_element_by_xpath("//div[@class='_3k-BhJ'][3]/table[1]/tbody/tr[2]/td[2]/ul/li")
        Smartphones['Processor'].append(processor.text)
    except NoSuchElementException:
        Smartphones['Processor'].append('-')
    
        
       
    # scraping processor core of smartphone
    try:
        core = driver.find_element_by_xpath("//div[@class='_3k-BhJ'][3]/table[1]/tbody/tr[3]/td[1]")
        if core.text != 'Processor Core' :
            core = driver.find_element_by_xpath("//div[@class='_3k-BhJ'][3]/table[1]/tbody/tr[2]/td[1]")
            if core.text != 'Processor Core' :
                raise NoSuchElementException
            else :
                cores = driver.find_element_by_xpath("//div[@class='_3k-BhJ'][3]/table[1]/tbody/tr[2]/td[2]/ul/li")
        else :
            cores = driver.find_element_by_xpath("//div[@class='_3k-BhJ'][3]/table[1]/tbody/tr[3]/td[2]/ul/li")
        Smartphones['Processor Cores'].append(disp_reso.text)
    except NoSuchElementException:
        Smartphones['Processor Cores'].append('-')
        
        
        
    # scraping the battery capacity of smartphone
    try:
        if driver.find_element_by_xpath("//div[@class='_3k-BhJ'][10]/div").text != "Battery & Power Features" :
            if driver.find_element_by_xpath("//div[@class='_3k-BhJ'][9]/div").text == "Battery & Power Features" :
                bat_tags = driver.find_element_by_xpath("//div[@class='_3k-BhJ'][9]/table/tbody/tr/td[1]")
                if bat_tags.text != "Battery Capacity" : raise NoSuchElementException
                bat_capa = driver.find_element_by_xpath("//div[@class='_3k-BhJ'][9]/table/tbody/tr/td[2]/ul/li")
            elif driver.find_element_by_xpath("//div[@class='_3k-BhJ'][8]/div").text == "Battery & Power Features" :
                bat_tags = driver.find_element_by_xpath("//div[@class='_3k-BhJ'][8]/table/tbody/tr/td[1]")
                if bat_tags.text != "Battery Capacity" : raise NoSuchElementException
                bat_capa = driver.find_element_by_xpath("//div[@class='_3k-BhJ'][8]/table/tbody/tr/td[2]/ul/li")
            else:
                raise NoSuchElementException
        else :
            bat_tags = driver.find_element_by_xpath("//div[@class='_3k-BhJ'][10]/table/tbody/tr/td[1]")
            if bat_tags.text != "Battery Capacity" : raise NoSuchElementException
            bat_capa = driver.find_element_by_xpath("//div[@class='_3k-BhJ'][10]/table/tbody/tr/td[2]/ul/li")
        Smartphones['Battery Capacity'].append(bat_capa.text)
    except NoSuchElementException:
        Smartphones['Battery Capacity'].append('-')
    
    
    
    
    # scraping price of smartphone
    try:
        price_tags = driver.find_element_by_xpath("//div[@class='_30jeq3 _16Jk6d']")
        Smartphones['Price'].append(price_tags.text)
    except NoSuchElementException:
          Smartphones['Price'].append('-') 


# In[27]:


print(len(Smartphones['Brand']),len(Smartphones['Phone name']), len(Smartphones['Colour']),len(Smartphones['RAM']),len(Smartphones['Storage(ROM)']),len(Smartphones['Primary Camera']),len(Smartphones['Secondary Camera']), len(Smartphones['Display Size']), len(Smartphones['Display Resolution']), len(Smartphones['Processor']), len(Smartphones['Processor Cores']), len(Smartphones['Battery Capacity']), len(Smartphones['Price']), len(Smartphones['URL'])) 


# In[28]:


df = pd.DataFrame.from_dict(Smartphones)
df


# In[29]:


# saving the data in csv
df.to_csv("smartphones.csv")


# In[30]:


url = 'https://www.google.co.in/maps'
driver.get(url)
time.sleep(2)


# In[31]:


City = input('Enter City name that has to be searched : ')
search_bar = driver.find_element_by_id('searchboxinput')
search_bar.click()
time.sleep(2)

#sending keys to find cities
search_bar.send_keys(City)

#checking for webelement and clicking on search button
search_btn = driver.find_element_by_id("searchbox-searchbutton")
search_btn.click()
time.sleep(2)

try:
    url_str = driver.current_url
    print("URL Extracted: ", url_str)
    latitude_longitude = re.findall(r'@(.*)data',url_str)
    if len(latitude_longitude):
        lat_lng_list = latitude_longitude[0].split(",")
        if len(lat_lng_list)>=2:
            latitude = lat_lng_list[0]
            longitude = lat_lng_list[1]
        print("Latitude = {}, Longitude = {}".format(latitude, longitude))
except Exception as e:
        print("Error: ", str(e))


# In[32]:


import pandas as pd
import selenium
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import StaleElementReferenceException, NoSuchElementException
import time


# In[33]:


driver=webdriver.Chrome(r'CC:\Users\Anu Shamaiah Prasad\Downloads\chromedriver_win32\chromedriver.exe')


# In[34]:


driver.get('https://www.digit.in')


# In[35]:


#finding the search button and clicking
search_laptop_button = driver.find_element_by_xpath('//div[@class="search"]')     
search_laptop_button.click()
search_bar=driver.find_element_by_id("globalPageSearchText")     
search_bar.send_keys('top 10 gaming laptop')
search_bar.send_keys(Keys.ENTER)


# In[36]:


#finding the search button and clicking
top10=driver.find_element_by_id("content_top10") 
top10.click()


# In[37]:


#finding the search button and clicking
best_laptop=driver.find_element_by_xpath("//*[text()='Best Gaming Laptops in India']") 
best_laptop.click()


# In[38]:


#creating empty list
Product=[]
Processor=[]
Memory=[]
OS=[]
Display=[]
Seller=[]
Price=[]
Availability=[]


# In[39]:


#scraping all the details
Brand_tag=driver.find_elements_by_xpath("//div[@class='right-container']/div/a/h3")
for i in Brand_tag:
    try:
        brandtag=i.text
        Product.append(brandtag)
    except NoSuchElementException   as e:
        Product.append('-')
        
specification = driver.find_elements_by_xpath('//div[@class="Specs-Wrap"]/ul')
for details in specification:
    lap_spec= details.find_elements_by_xpath('.//div[@class="value"]')
    try:
        OS.append(lap_spec[0].text)
        Display.append(lap_spec[1].text)
        Processor.append(lap_spec[2].text)
        Memory.append(lap_spec[3].text)
    except NoSuchElementException   as e:
        OS.append('-')
        Displah.append('-')
        Processor.append('-')
        Memory.append('-')
        
seller_info= driver.find_elements_by_xpath('//td[@class="smmerchant"]')
for info in seller_info:
    try:
        Seller.append(info.text)
    except NoSuchElementException   as e:
        Seller.append('-')
    
price = driver.find_elements_by_xpath('//td[@class="smprice"]')
for prices in price:
    try:
        Price.append(prices.text)
    except NoSuchElementException   as e:
        Price.append('-') 
    
Availability_tag=driver.find_elements_by_xpath("//p[@style='margin: 0px 0 0 0;font-size: 13px;position: relative;font-weight: 400;line-height: 19px;width: 89px;']")
for i in Availability_tag:
    try:
        Availabilitytag1=i.text
        Availability.append(Availabilitytag1.replace('\n',""))
    except NoSuchElementException   as e:
        Availability.append('-')


# In[40]:


#creating the dataset
Top_10_Gaming_Laptop=pd.DataFrame({})
Top_10_Gaming_Laptop['Product']=Product[:10]
Top_10_Gaming_Laptop['Processor']=Processor[:10]
Top_10_Gaming_Laptop['Memory']=Memory[:10]
Top_10_Gaming_Laptop['OS']=OS[:10]
Top_10_Gaming_Laptop['Display']=Display[:10]
Top_10_Gaming_Laptop['Seller']=Seller[:10]
Top_10_Gaming_Laptop['Price']=Price[:10]
Top_10_Gaming_Laptop['Availability']=Availability[:10]


# In[41]:


import pandas as pd
import requests

headers = {
    "accept": "application/json, text/plain, */*",
    "referer": "https://www.forbes.com/global2000/",
    "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.67 Safari/537.36",
}

cookies = {
    "notice_behavior": "expressed,eu",
    "notice_gdpr_prefs": "0,1,2:1a8b5228dd7ff0717196863a5d28ce6c",
}

api_url = "https://www.forbes.com/forbesapi/org/global2000/2020/position/true.json?limit=2000"
response = requests.get(api_url, headers=headers, cookies=cookies).json()

sample_table = [
    [
        item["organizationName"],
        item["country"],
        item["revenue"],
        item["profits"],
        item["assets"],
        item["marketValue"]
    ] for item in
    sorted(response["organizationList"]["organizationsLists"], key=lambda k: k["position"])
]

df = pd.DataFrame(sample_table, columns=["Company", "Country", "Sales", "Profits", "Assets", "Market Value"])
df.to_csv("forbes_2020.csv", index=False)


# In[42]:


import pandas as pd
import requests

headers = {
    "accept": "application/json, text/plain, */*",
    "user-agent": "Mozilla/5.0 (X11; Linux x86_64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/87.0.4280.67 Safari/537.36",
}

cookies = {
    "notice_behavior": "expressed,eu",
    "notice_gdpr_prefs": "0,1,2:1a8b5228dd7ff0717196863a5d28ce6c",
}

year = 2019
api_url = f"https://www.forbes.com/forbesapi/org/global2000/{year}/position/true.json?limit=2000"
response = requests.get(api_url, headers=headers, cookies=cookies).json()
df = pd.DataFrame(
    sorted(
        response["organizationList"]["organizationsLists"],
        key=lambda k: k["position"],
    )
)
df.to_csv("forbes_2019.csv", index=False)


# In[44]:


from selenium import webdriver
import time

# Set up the WebDriver
driver = webdriver.Chrome((r"C:\Users\Anu Shamaiah Prasad\Downloads\chromedriver_win32\chromedriver.exe"))  # Replace with the path to your WebDriver executable

# Open the YouTube video
video_url = 'https://www.youtube.com/watch?v=your_video_id'  # Replace with the URL of the YouTube video
driver.get(video_url)

# Scroll to load comments
scroll_pause_time = 2  # Adjust the pause time as needed
scrolls = 10  # Adjust the number of scrolls as needed

for _ in range(scrolls):
  driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
  time.sleep(scroll_pause_time)

# Extract comments, upvotes, and time
comments = driver.find_elements_by_xpath('//yt-formatted-string[@id="content-text"]')
upvotes = driver.find_elements_by_xpath('//span[@id="vote-count-middle"]')
times = driver.find_elements_by_xpath('//a[@class="yt-simple-endpoint style-scope yt-formatted-string"]')

# Store the extracted data
extracted_data = []
for comment, upvote, time in zip(comments, upvotes, times):
  extracted_data.append({
  'comment': comment.text,
  'upvote': upvote.text,
  'time': time.text
  })

# Close the WebDriver
driver.quit()

# Print the extracted data
for data in extracted_data:
  print(data)


# In[45]:


# getting the web page of mentioned url
url = "https://www.hostelworld.com/"
driver.get(url)
time.sleep(3)


# In[46]:


# locating the location search bar
search_bar = driver.find_element_by_id("search-input-field")

# entering London in search bar
search_bar.send_keys("London")


# In[47]:


# select London
London = driver.find_element_by_xpath("//ul[@id='predicted-search-results']//li[2]")
#clicking on button
London.click()

# do click on Let's Go button
search_btn = driver.find_element_by_id('search-button')
search_btn.click()


# In[48]:


# creating empty list & find required data
hostel_name = []
distance = []
pvt_prices = []
dorms_price = []
rating = []
reviews = []
over_all = []
facilities = []
description = []
url = []


# In[49]:


# scraping the required informations
for i in driver.find_elements_by_xpath("//div[@class='pagination-item pagination-current' or @class='pagination-item']"):
    i.click()
    time.sleep(3)
    
    
    # scraping  hostel name
    try:
        name = driver.find_elements_by_xpath("//h2[@class='title title-6']")
        for i in name:
            hostel_name.append(i.text)
    except NoSuchElementException:
        hostel_name.append('-')
        
        
    # scraping distance from city centre
    try:
        dist = driver.find_elements_by_xpath("//div[@class='subtitle body-3']//a//span[1]")
        for i in name:
            distance.append(i.text.replace('Hostel - ',''))
    except NoSuchElementException:
        distance.append('-')
        
   
    for i in driver.find_elements_by_xpath("//div[@class='prices-col']"):   
    # scraping privates from price
        try:
            pvt_price = driver.find_element_by_xpath("//a[@class='prices']//div[1]//div")
            pvt_prices.append(pvt_price.text)
        except NoSuchElementException:
            pvt_prices.append('-')
   

    for i in driver.find_elements_by_xpath("//div[@class='prices-col']"):          
    # scraping dorms from price
        try:
            dorms = driver.find_element_by_xpath("//a[@class='prices']//div[2]/div")
            dorms_price.append(dorms.text)
        except NoSuchElementException:
            dorms_price.append('-')
            
            
    # scraping facilities
    try:
        fac1 = driver.find_elements_by_xpath("//div[@class='has-wifi']")
        fac2 = driver.find_elements_by_xpath("//div[@class='has-sanitation']")
        for i in fac1:
            for j in fac2:
                facilities.append(i.text +', '+ j.text)
    except NoSuchElementException:
        facilities.append('-')
     
            
    #fetching url of each hostel
    p_url = driver.find_elements_by_xpath("//div[@class='prices-col']//a[2]")
    for i in p_url:
        url.append(i.get_attribute("href"))
        
for i in url:
    driver.get(i)
    time.sleep(3)
    

    # scraping ratings
    try:
        rat = driver.find_element_by_xpath("//div[@class='score orange big' or @class='score gray big']")
        rating.append(rat.text)
    except NoSuchElementException:
        rating.append('-')
        
        
    # scraping total review
    try:
        rws = driver.find_element_by_xpath("//div[@class='reviews']")
        reviews.append(rws.text.replace('Total Reviews',''))
    except NoSuchElementException:
        reviews.append('-')
        
        
    # fetching over all review
    try:
        overall = driver.find_element_by_xpath("//div[@class='keyword']//span")
        over_all.append(overall.text)
    except NoSuchElementException:
        over_all.append('-')
        
        
    # fetching property description
    try:
        disc = driver.find_element_by_xpath("//div[@class='content']")
        description.append(disc.text)
    except NoSuchElementException:
        over_all.append('-')
    
    # do click on show more button for description
    try:
        driver.find_element_by_xpath("//a[@class='toggle-content']").click()
        time.sleep(4)
    except NoSuchElementException:
        pass


# In[50]:


print(len(hostel_name),
len(distance),
len(pvt_prices),
len(dorms_price),
len(rating),
len(reviews),
len(over_all),
len(facilities),
len(description),
len(url))


# In[51]:


# creating DataFrame
Hostel = pd.DataFrame({})
Hostel['Hostel Name'] = hostel_name
Hostel['Distance from City Centre'] = distance
Hostel['Ratings'] = rating
Hostel['Total Reviews'] = reviews
Hostel['Overall Reviews'] = over_all
Hostel['Privates from Price'] = pvt_prices
Hostel['Dorms from Price'] = dorms_price
Hostel['Facilities'] = facilities[:74]
Hostel['Description'] = description
Hostel


# In[52]:


# saving the dataset to csv
Hostel.to_csv("London_Hostels.csv")


# In[ ]:




