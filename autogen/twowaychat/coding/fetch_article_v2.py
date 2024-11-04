# filename: fetch_article_v2.py
import requests
from bs4 import BeautifulSoup

url = "https://microsoft.github.io/autogen/0.2/blog/2024/03/03/AutoGen-Update/"
response = requests.get(url)
if response.status_code == 200:
    soup = BeautifulSoup(response.content, 'html.parser')
    article_div = soup.find('div', class_='content')
    if article_div:
        article_content = article_div.get_text()
        print(article_content)
    else:
        print("Could not find the article content.")
else:
    print("Unable to retrieve the article content.")