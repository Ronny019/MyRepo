from bs4 import BeautifulSoup
import requests
html = requests.get("https://www.oreilly.com/terms/").text
soup = BeautifulSoup(html, 'html5lib')
first_paragraph = soup.find('p') # or just soup.p
first_paragraph_text = soup.p.text
first_paragraph_words = soup.p.text.split()

print(first_paragraph_words)

#first_paragraph_id = soup.p['id'] # raises KeyError if no 'id'
first_paragraph_id2 = soup.p.get('id') # returns None if no 'id'

print(first_paragraph_id2)

all_paragraphs = soup.find_all('p') # or just soup('p')
paragraphs_with_ids = [p for p in soup('p')]

#for paragraph in paragraphs_with_ids:
#    print(paragraph.text)

important_paragraphs = soup('p', {'class' : 'important'})
important_paragraphs2 = soup('p', 'important')
important_paragraphs3 = [p for p in soup('p')
                        if 'important' in p.get('class', [])]

spans_inside_divs = [span
                    for div in soup('div') # for each <div> on the page
                    for span in div('span')] # find each <span> inside it

for span in spans_inside_divs:
    print(span.text)
