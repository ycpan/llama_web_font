from serpapi import GoogleSearch
search = GoogleSearch({
    "q": "新能源产业", 
    "location": "Austin,Texas",
    "api_key": "AIzaSyBUWJHf8EFD8O_72OHgLo2n0VZDIg2iFIQ"
  })
result = search.get_dict()
print(result)
