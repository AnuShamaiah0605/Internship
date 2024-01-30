#!/usr/bin/env python
# coding: utf-8

# In[1]:


import selenium
import pandas as pd
from selenium import webdriver
driver = webdriver.Chrome(r"C:\Users\Anu Shamaiah Prasad\Downloads\chromedriver_win32\chromedriver.exe")
url = "https://www.naukri.com"
driver.get(url)
search_job = driver.find_element_by_xpath("//input[@class='sugInp']")
search_job
search_job.send_keys('Data Analyst')
search_loc=driver.find_element_by_id('qsb-location-sugg')
search_loc.send_keys("Bangalore")
search_btn= driver .find_element_by_xpath("//button[@class='btn']")
search_btn
search_btn=driver.find_element_by_xpath("//button[@class='btn']")
search_btn.click()
title_tags=driver.find_elements_by_xpath("//a[@class='title fw500 ellipsis']")
title_tags
job_titles=[]
for i in title_tags:
    if i.text is None:
        job_titles.append('Not')
    else:
        job_titles.append(i.text)
job_titles[:10]  
company_tags=driver.find_elements_by_xpath("//a[@class='subTitle ellipsis fleft']")
company_tags
companies_names=[]

for i in company_tags:
    companies_names.append(i.text)
companies_names[:10]  
experience_tags=driver.find_elements_by_xpath("//li[@class='fleft grey-text br2 placeHolderLi experience'] //span")
experience_tags
experience_list=[]
for i in experience_tags:
    experience_list.append(i.text)
experience_list[:10] 
locations_tags=driver.find_elements_by_xpath("//li[@class='fleft grey-text br2 placeHolderLi location']/span")
locations_tags
locations_list=[]
for i in locations_tags:
    locations_list.append(i.text)
locations_list[:10] 
print(len(job_titles[:10])),print(len(companies_names[:10])),print(len(experience_list[:10])),print(len(locations_l))


# In[2]:


jobs=pd.DataFrame({})
jobs['title']=job_titles[:10]
jobs['company']=companies_names[:10]
jobs['experience_required']=experience_list[:10]
jobs['location']=locations_list[:10]


# In[3]:


jobs


# In[4]:


driver.close()


# In[5]:


from selenium.common.exceptions import NoSuchElementException


# In[6]:


search_btn=driver.find_element_by_class_name('btn')
search_btn.click()


# In[7]:


title_tag=driver.find_elements_by_xpath("//a[@class='title fw500 ellipsis']")
title_tag
job1_titles=[]
for i in title_tag:
    if i.text is None:
        job1_titles.append('Not')
    else:
        job1_titles.append(i.text)
job1_titles[:10] 
company_tag=driver.find_elements_by_xpath("//a[@class='subTitle ellipsis fleft']")
company_tag
companies1_names=[]

for i in company_tag:
    companies1_names.append(i.text)
companies1_names[:10]   
locations_tag=driver.find_elements_by_xpath("//li[@class='fleft grey-text br2 placeHolderLi location']/span")
locations_tag
locations1_list=[]
for i in locations_tag:
    locations1_list.append(i.text)
locations1_list[:10]   
print(len(job1_titles[:10])),print(len(companies1_names[:10])),print(len(locations1_list[:10]))
driver=webdriver.Chrome(r"C:\Users\Neha\Downloads\chromedriver_win32\chromedriver.exe")
driver.get('https://www.naukri.com/data-scientist-jobs-in-banglore-bagaluru')
urls=[]
job_description=[]
for i in driver.find_elements_by_xpath("//a[@class='title fw500 ellipsis']"):
    urls.append(i.get_attribute("href"))
for url in urls[:10]:
    
    
    try:
        driver.get(url)
        description=driver.find_element_by_xpath("//section[@class='job-desc']").text
        job_description.append(description)
        
    except NoSuchElementException:
        job_description.append("Not Available")
job_description
print(len(job_description))


# In[8]:


job1=pd.DataFrame({})
job1['title']=job1_titles[:10]
job1['company_name']=companies1_names[:10]
job1['location']=locations1_list[:10]
job1['job_desc']=job_description


# In[9]:


job1


# In[10]:


company_t1=driver.find_elements_by_xpath("//a[@class='subTitle ellipsis fleft']")
company_t1
companies_names=[]

for i in company_t1:
    companies_names.append(i.text)
companies_names[:10]   
experience_t1=driver.find_elements_by_xpath("//li[@class='fleft grey-text br2 placeHolderLi experience'] //span")
experience_t1
experience_list=[]
for i in experience_t1:
    experience_list.append(i.text)
experience_list[:10]   
locations_t1=driver.find_elements_by_xpath("//li[@class='fleft grey-text br2 placeHolderLi location']/span")
locations_t1
locations_list=[]
for i in locations_t1:
    locations_list.append(i.text)
locations_list[:10]   
print(len(job_titles[:10])),print(len(companies_names[:10])),print(len(experience_list[:10])),print(len(locations_l))
jobs2=pd.DataFrame({})
jobs2['title']=job_titles[:10]
jobs2['company']=companies_names[:10]
jobs2['experience_required']=experience_list[:10]
jobs2['location']=locations_list[:10]


jobs2


# In[11]:


import requests
page=requests.get("https://www.glassdoor.co.in/index.htm")
page


# In[12]:


pag1 =requests.get("https://www.glassdoor.co.in/Salaries/index.htm")
pag1


# In[13]:


url="https://www.flipkart.com/"
driver.get(url)
search_g= driver.find_element_by_xpath("//input[@type='text']")
search_g
search_g.send_keys('sunglasses')
search_btn=driver.find_element_by_xpath("//button[@class='L0Z3Pu']")
search_btn
search_btn=driver.find_element_by_class_name('L0Z3Pu')
search_btn.click()
B_name=[]
Price=[]
P_desc=[]
Discount=[]
for i in range(3):
    b_name=driver.find_elements_by_xpath("//div[@class='_2WkVRV']")
    p_desc=driver.find_elements_by_xpath("//a[@class='IRpwTa']")
    price =driver.find_elements_by_xpath("//div[@class='_25b18c']")
    discount=driver.find_elements_by_xpath("//div[@class='_3Ay6Sb']")
    
    for j  in b_name:
        B_name.append(j.text)
    B_name[:100]    
    
    
    
    for k in p_desc:
        P_desc.append(k.text)
    P_desc[:100] 
    
    
    for l in price:
        Price.append(l.text)
    Price[:100] 
    
    
    for t in discount:
        Discount.append(t.text)
    Discount[:100]
B_name[:100]


# In[14]:


print(len(B_name[:100])),print(len(Price[:100])),print(len(P_desc[:100])),print(len(Discount[:100]))


# In[15]:


sun_gl=pd.DataFrame({})
sun_gl['Brand_name']=B_name[:100]
sun_gl['P_price']=Price[:100]
sun_gl['Pr_desc']=P_desc[:100]
sun_gl['P_discount']=Discount[:100]

sun_gl


# In[16]:


search_g.send_keys('sneakers')
search_btn=driver.find_element_by_xpath("//button[@class='L0Z3Pu']")
search_btn
search_btn=driver.find_element_by_class_name('L0Z3Pu')
search_btn.click()
B_name=[]
Price=[]
P_desc=[]
Discount=[]
for i in range(3):
    b_name=driver.find_elements_by_xpath("//div[@class='_2WkVRV']")
    p_desc=driver.find_elements_by_xpath("//a[@class='IRpwTa']")
    price =driver.find_elements_by_xpath("//div[@class='_25b18c']")
    discount=driver.find_elements_by_xpath("//div[@class='_3Ay6Sb']")
    
    for j  in b_name:
        B_name.append(j.text)
    B_name[:100]    
    
    
    
    for k in p_desc:
        P_desc.append(k.text)
    P_desc[:100] 
    
    
    for l in price:
        Price.append(l.text)
    Price[:100] 
    
    
    for t in discount:
        Discount.append(t.text)
    Discount[:100]

print(len(B_name[:100])),print(len(Price[:100])),print(len(P_desc[:100])),print(len(Discount[:100]))


# In[17]:


sun_gl=pd.DataFrame({})
sun_gl['Brand_name']=B_name[:100]
sun_gl['P_price']=Price[:100]
sun_gl['Pr_desc']=P_desc[:100]
sun_gl['P_discount']=Discount[:100]

sun_gl


# In[18]:


url="https://www.myntra.com/shoes"
driver.get(url)
filter_button=driver.find_elements_by_xpath("//label[@class='common-customerCheckbox vertical-filters-label']")
for i in filter_button:
    if i.text=="Rs. 6649 to Rs. 13099":
        i.click()
        break
#Applying the black colur filter
filter_button=driver.find_elements_by_xpath("//li[@class='colour-listItem']")
for i in filter_button:
    if i.text=="Black":
        i.click()
        break
B_name=[]
Price=[]
P_desc=[]

for i in range(2):
    b_name=driver.find_elements_by_xpath("//h1[@class='pdp-title']")
    p_desc=driver.find_elements_by_xpath("//h1[@class='name']")
    price =driver.find_elements_by_xpath("//div[@class='product-price']")
 
    for j  in b_name:
        B_name.append(j.text)
    B_name[:100]    
    
    
    
    for k in p_desc:
        P_desc.append(k.text)
    P_desc[:100] 
    
    
    for l in price:
        Price.append(l.text)
    Price[:100] 
    

print(len(B_name[:100])),print(len(Price[:100])),print(len(P_desc[:100]))


# In[ ]:




