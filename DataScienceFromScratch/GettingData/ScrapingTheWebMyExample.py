from bs4 import BeautifulSoup
import requests
html = requests.get("https://iwriteforyousweetheart.wordpress.com/").text
soup = BeautifulSoup(html, 'html5lib')

article = soup.find('article',{'id':'post-3207'})
header = article.find('header').find('h1').find('a').text
print(header)
time = article.find('div',{'class':'entry-meta'}).find('time',{'class':'updated'}).text
print(time)
entry_content = article.find('div',{'class':'entry-body'})
article_lines = entry_content.find_all('div',{'class':'_1mf _1mj'})
article_body = ''
for line in article_lines:
    spans = line.find_all('span')
    for span in spans:
        article_body+=span.text
    article_body+='\n'
print(article_body)