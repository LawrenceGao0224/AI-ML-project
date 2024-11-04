# filename: fetch_article_v4.py
import requests
from bs4 import BeautifulSoup

url = "https://microsoft.github.io/autogen/0.2/blog/2024/03/03/AutoGen-Update/"
response = requests.get(url)
if response.status_code == 200:
    soup = BeautifulSoup(response.content, 'html.parser')
    article_content = soup.find('div', class_='entry-content')
    if article_content:
        paragraphs = article_content.find_all('p')
        article_text = "\n".join([p.get_text() for p in paragraphs])
        print(article_text)
    else:
        print("Could not find the article content.")
else:
    print("Unable to retrieve the article content.")