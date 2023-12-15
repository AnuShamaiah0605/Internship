#!/usr/bin/env python
# coding: utf-8

# In[5]:


"Write a Python program to replace all occurrences of a space, comma, or dot with a colon."
import pandas as pd
import re

str = 'Python Exercises, PHP exercises.'
data = re.sub("[ ,.]",":",str)
print(data)


# In[3]:


"Create a dataframe using the dictionary below and remove everything (commas (,), !, XXXX, ;, etc.) from the columns except words"


# In[15]:


str = {'SUMMARY' : ['hello, world!', 'XXXXX test', '123four, five:; six...']}
data =pd.DataFrame(str)
data['']=data['SUMMARY'].str.replace('[^a-zA-Z\s]','',regex=True)
print(data)


# In[21]:


str = {'SUMMARY' : ['hello, world!', 'XXXXX test', '123four, five:; six...']}
data =pd.DataFrame(str)
data['SUMMARY']=data['SUMMARY'].str.replace('[^a-zA-Z\s]','',regex=True)
print(data)


# the str replace() function in the pandas library is employed to eliminate all characters in the 'SUMMARY' column that are not alphabets or spaces. The regular expression [^a-zA-Z\s] is utilized to identify any character that does not fall within the range of alphabets or spaces. This identified character is then substituted with an empty string, resulting in its removal from the column.

# In[42]:


"Create a function in python to find all three, four, and five character words in a string.The use of the re.compile() method is mandatory"

def find_words(string):
  pattern = re.compile(r'\b\w{3,5}\b')
  matches = pattern.findall(string)
  return matches

string = "this function with a string to find all words that are at least three, four, or five characters long"
result = find_words(string)
print(result)


# In[43]:


"Write a python program to remove the parenthesis area from the text stored in the text file using Regular Expression"

def remove_parentheses(strings):
  pattern = re.compile(r'\(\)')
  modified_strings = []
  for string in strings:
    modified_string = re.sub(r" ?\([^)]+\)", '', string)
    modified_strings.append(modified_string)
  return modified_strings
sample_text = ["example (.com)", "hr@fliprobo (.com)", "github (.com)", "Hello (Data Science World)", "Data (Scientist)"]
result = remove_parentheses(sample_text)
print(result)


# In[44]:


"Create a function in Python to remove the parenthesis in a list of strings. The use of the re.compile() method is mandatory."
def remove_parentheses(strings):
  pattern = re.compile(r'\(\)')
  modified_strings = []
  for string in strings:
    modified_string = re.sub( pattern , '', string)
    modified_strings.append(modified_string)
  return modified_strings
sample_text = ["example (.com)", "hr@fliprobo (.com)", "github (.com)", "Hello (Data Science World)", "Data (Scientist)"]
result = remove_parentheses(sample_text)
print(result)


# In[45]:


"Write a regular expression in Python to split a string into uppercase letters."
text = "ImportanceOfRegularExpressionsInPython"
print(re.findall('[A-Z][^A-Z]*', text))


# In[46]:


"Create a function in python to insert spaces between words starting with numbers"
def insert_spaces(text):
  # Use regular expression to find words starting with numbers
  pattern = r'(\d+)([A-Za-z]+)'
  result = re.sub(pattern, r'\1 \2', text)
  return result
text = "RegularExpression1IsAn2ImportantTopic3InPython"
output = insert_spaces(text)
print(output)


# In[47]:


"- Create a function in python to insert spaces between words starting with capital letters or with numbers."
def insert_spaces(text):
  # Use regular expression to find words starting with numbers
  pattern = r'(\d+)([A-Za-z]+)'
  result = re.sub(pattern, r' \1 \2', text)
  return result
text = "RegularExpression1IsAn2ImportantTopic3InPython"
output = insert_spaces(text)
print(output)


# In[49]:


"Write a Python program to match a string that contains only upper and lowercase letters, numbers, and underscores."

def match_string(string):
  pattern = r'^[a-zA-Z0-9_]+$'
  if re.match(pattern, string):
   print("String matches the pattern")
  else:
   print("String does not match the pattern")

# Example usage
match_string("Hello_World123")  # Output: String matches the pattern
match_string("Hello World")    # Output: String does not match the pattern


# In[54]:


"Write a Python program where a string will start with a specific number"
def match_string(string):
  pattern = r'^[a-zA-Z0-9_]+$'
  if re.match(pattern, string):
   print("String matches the pattern")
  else:
   print("String does not match the pattern")

# Example usage
match_string("Hello_World123")  # Output: String matches the pattern
match_string("Hello World")


# In[55]:


"Write a Python program to remove leading zeros from an IP address"
ip = "216.08.094.196"
string = re.sub('\.[0]*', '.', ip)
print(string)


# In[57]:


"Write a regular expression in python to match a date string in the form of Month name followed by day number and year stored in a text file."
text = "On August 15th 1947 that India was declared independent from British colonialism, and the reins of control were handed over to the leaders of the Country."

pattern = r"\b([A-Z][a-z]+) \d{1,2}(?:st|nd|rd|th)? \d{4}\b"

matches = re.findall(pattern, text)
print(matches)


# In[1]:


f = open("C:\Users\Anu Shamaiah Prasad\OneDrive\Desktop\doc.txt", "r")

content = f.read()

#The regex pattern that we created
pattern = "\d{2}[/-]\d{2}[/-]\d{4}"

#Will return all the strings that are matched
dates = re.findall(pattern, content)


# In[3]:


"Write a Python program to search some literals strings in a string. "
import re
patterns = [ 'fox', 'dog', 'horse' ]
text = 'The quick brown fox jumps over the lazy dog.'
for pattern in patterns:
    print('Searching for "%s" in "%s" ->' % (pattern, text),)
    if re.search(pattern,  text):
        print('Matched!')
    else:
        print('Not Matched!')


# In[4]:


"Write a Python program to search a literals string in a string and also find the location within the original string where the pattern occurs"
import re
pattern = 'fox'
text = 'The quick brown fox jumps over the lazy dog.'
match = re.search(pattern, text)
s = match.start()
e = match.end()
print('Found "%s" in "%s" from %d to %d ' %     (match.re.pattern, match.string, s, e))


# In[5]:


"Write a Python program to find the substrings within a string."
text = 'Python exercises, PHP exercises, C# exercises'
pattern = 'exercises'
for match in re.findall(pattern, text):
    print('Found "%s"' % match)


# In[6]:


"Write a Python program to find the occurrence and position of the substrings within a string."
text = 'Python exercises, PHP exercises, C# exercises'
pattern = 'exercises'
for match in re.finditer(pattern, text):
    s = match.start()
    e = match.end()
    print('Found "%s" at %d:%d' % (text[s:e], s, e))


# In[7]:


"Write a Python program to convert a date of yyyy-mm-dd format to dd-mm-yyyy format."
def change_date_format(dt):
        return re.sub(r'(\d{4})-(\d{1,2})-(\d{1,2})', '\\3-\\2-\\1', dt)
dt1 = "2026-01-02"
print("Original date in YYY-MM-DD Format: ",dt1)
print("New date in DD-MM-YYYY Format: ",change_date_format(dt1))


# In[9]:


"Create a function in python to find all decimal numbers with a precision of 1 or 2 in a string. The use of the re.compile() method is mandatory."
def find_decimal_numbers(string):
  pattern = re.compile(r'\d+\.\d{1,2}')
  decimal_numbers = re.findall(pattern, string)
  return decimal_numbers
sample_text = "01.12 0132.123 2.31875 145.8 3.01 27.25 0.25"
output = find_decimal_numbers(sample_text)
print(output)


# In[11]:


"Write a Python program to separate and print the numbers and their position of a given string."
text = "The following example creates an ArrayList with a capacity of 50 elements. 4 elements are then added to the ArrayList and the ArrayList is trimmed accordingly."

for m in re.finditer("\d+", text):
    print(m.group(0))
    print("Index position:", m.start())


# In[13]:


"Write a regular expression in python program to extract maximum/largest numeric value from a string."
input_string = 'My marks in each semester are: 947, 896, 926, 524, 734, 950, 642,1100'

numeric_values = re.findall(r'\d+', input_string)
numeric_values = [int(value) for value in numeric_values]

max_value = max(numeric_values)

print(max_value)


# In[17]:


"Create a function in python to insert spaces between words starting with capital letters."
def capital_words_spaces(str1):
  return re.sub(r"(\w)([A-Z])", r"\1 \2", str1)

print(capital_words_spaces("RegularExpressionIsAnImportantTopicInPython"))


# In[18]:


"Python regex to find sequences of one upper case letter followed by lower case letters"
def text_match(text):
        patterns = '[A-Z]+[a-z]+$'
        if re.search(patterns, text):
                return 'Found a match!'
        else:
                return('Not matched!')
print(text_match("AaBbGg"))
print(text_match("Python"))
print(text_match("python"))
print(text_match("PYTHON"))
print(text_match("aA"))
print(text_match("Aa"))


# In[19]:


"Write a Python program to remove continuous duplicate words from Sentence using Regular Expression"
def unique_list(text_str):
    # Split the input string into a list of words
    l = text_str.split()
    
    # Initialize an empty list to store unique words
    temp = []
    
    # Iterate through each word in the list
    for x in l:
        # Check if the word is not already in the temporary list
        if x not in temp:
            # If true, add the word to the temporary list
            temp.append(x)
    
    # Join the unique words into a string and return the result
    return ' '.join(temp)

# Initialize a string
text_str = "Hello hello world world"

# Print the original string
print("Original String:")
print(text_str)

# Print a newline for better formatting
print("\nAfter removing duplicate words from the said string:")

# Call the function to remove duplicate words and print the result
print(unique_list(text_str)) 


# In[21]:


"Write a python program using RegEx to accept string ending with alphanumeric character."
def check_string(string):
  pattern = r"\w$"
  match = re.search(pattern, string)
  if match:
   return True
  else:
   return False

# Example usage
input_string = input("Enter a string: ")
if check_string(input_string):
  print("String ends with an alphanumeric character")
else:
  print("String does not end with an alphanumeric character")


# In[22]:


"Write a python program using RegEx to extract the hashtags"
def extract_hashtags(text):
  hashtags = re.findall(r'#\w+', text)
  return hashtags

# Sample text
text = 'RT @kapil_kausik: #Doltiwal I mean #xyzabc is "hurt" by #Demonetization as the same has rendered USELESS <ed><U+00A0><U+00BD><ed><U+00B1><U+0089> "acquired funds" No wo'

# Extract hashtags
hashtags = extract_hashtags(text)

# Print the extracted hashtags
print(hashtags)


# In[23]:


"Write a python program using RegEx to remove <U+..> like symbols"
input_text = "@Jags123456 Bharat band on 28??<ed><U+00A0><U+00BD><ed><U+00B8><U+0082>Those who are protesting #demonetization are all different party leaders"

pattern = r"<U\+\w{4}>"
output_text = re.sub(pattern, "", input_text)

print(output_text)


# In[24]:


"Write a python program to extract dates from the text stored in the text file"
with open('filename.txt', 'r') as file:
  text = file.read()

# Define the regular expression pattern for dates
pattern = r'\d{2}-\d{2}-\d{4}'

# Find all matches of the pattern in the text
dates = re.findall(pattern, text)

# Print the extracted dates
for date in dates:
  print(date)


# In[26]:


""
def remove_words(string):
  pattern = re.compile(r'\b\w{2,4}\b')
  modified_string = re.sub(pattern, '', string)
  return modified_string

sample_text = "The following example creates an ArrayList with a capacity of 50 elements. 4 elements are then added to the ArrayList and the ArrayList is trimmed accordingly."
output = remove_words(sample_text)
print(output) 


# In[ ]:




