# filename: fetch_article_v5.py
import requests
from bs4 import BeautifulSoup

url = "https://microsoft.github.io/autogen/0.2/blog/2024/03/03/AutoGen-Update/"
response = requests.get(url)
if response.status_code == 200:
    soup = BeautifulSoup(response.content, 'html.parser')
    main_text = soup.find('body').get_text()
    print(main_text)
else:
    print("Unable to retrieve the article content.")