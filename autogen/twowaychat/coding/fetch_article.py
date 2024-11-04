# filename: fetch_article.py
import requests
from bs4 import BeautifulSoup

url = "https://microsoft.github.io/autogen/0.2/blog/2024/03/03/AutoGen-Update/"
response = requests.get(url)
if response.status_code == 200:
    soup = BeautifulSoup(response.content, 'html.parser')
    article_content = soup.find('div', class_='content').get_text()
    print(article_content)
else:
    print("Unable to retrieve the article content.")