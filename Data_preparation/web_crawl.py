import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

class WebCrawler:
    def __init__(self, base_url, max_depth=2):
        self.base_url = base_url
        self.max_depth = max_depth
        self.visited = set()
        self.results = []

    def fetch_page(self, url):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            print(f"Failed to fetch {url}: {e}")
            return None

    def extract_links(self, url, html):
        soup = BeautifulSoup(html, 'html.parser')
        links = set()
        for tag in soup.find_all('a', href=True):
            href = urljoin(url, tag['href'])
            parsed = urlparse(href)
            if parsed.netloc and parsed.scheme in ('http', 'https'):
                links.add(href)
        return links

    def crawl(self, url, depth):
        if depth > self.max_depth or url in self.visited:
            return

        self.visited.add(url)
        print(f"Crawling: {url} (Depth: {depth})")
        html = self.fetch_page(url)
        if html is None:
            return

        self.results.append((url, html))

        links = self.extract_links(url, html)
        for link in links:
            self.crawl(link, depth + 1)

    def save_results_to_file(self, urls_filename, content_filename):
        try:
            with open(urls_filename, 'w') as urls_file:
                for url, _ in self.results:
                    urls_file.write(url + '\n')

            with open(content_filename, 'w') as content_file:
                for url, html in self.results:
                    content_file.write(f"URL: {url}\n")
                    content_file.write(f"Content:\n{html}\n\n")

            print(f"Results saved to {urls_filename} and {content_filename}")
        except IOError as e:
            print(f"Failed to save results: {e}")

if __name__ == "__main__":
    # Example usage
    base_url = "https://medium.com/seaniap/python%E7%88%AC%E8%9F%B2-%E7%B6%B2%E8%B7%AF%E6%95%B8%E6%93%9A%E8%B3%87%E6%96%99%E7%9A%84%E7%88%AC%E5%8F%96%E6%8A%80%E5%B7%A7-1-69c267b33273"  # Replace with the target URL
    crawler = WebCrawler(base_url, max_depth=0)

    crawler.crawl(crawler.base_url, depth=0)

    print("\nCrawled URLs:")
    for url, _ in crawler.results:
        print(url)

    # Save results to files
    urls_file = "crawled_urls.txt"
    content_file = "crawled_content.txt"
    crawler.save_results_to_file(urls_file, content_file)
