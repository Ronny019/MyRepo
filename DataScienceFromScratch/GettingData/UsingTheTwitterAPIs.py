# Create the following files in the directory "C:\Twitter API" with appropriate information
# Access Token, Access Token Secret, API Key, API Key Secret , App Name

from twython import Twython

CONSUMER_KEY = ''
CONSUMER_SECRET = ''
ACCESS_TOKEN = ''
ACCESS_TOKEN_SECRET = ''
with open('C:\Twitter API\API Key.txt','r') as key:
    for line in key:
        CONSUMER_KEY = line

with open('C:\Twitter API\API Key Secret.txt','r') as secret:
    for line in secret:
        CONSUMER_SECRET = line

with open('C:\Twitter API\Access Token.txt','r') as key:
    for line in key:
        ACCESS_TOKEN = line

with open('C:\Twitter API\Access Token Secret.txt','r') as secret:
    for line in secret:
        ACCESS_TOKEN_SECRET = line

twitter = Twython(CONSUMER_KEY, CONSUMER_SECRET)

# search for tweets containing the phrase "data science"
for status in twitter.search(q='"data science"')["statuses"]:
    user = status["user"]["screen_name"].encode('utf-8')
    text = status["text"].encode('utf-8')
    print (user, ":", text)
    print ()



from twython import TwythonStreamer
# appending data to a global variable is pretty poor form
# but it makes the example much simpler
tweets = []
class MyStreamer(TwythonStreamer):
    """our own subclass of TwythonStreamer that specifies
    how to interact with the stream"""
    def on_success(self, data):
        """what do we do when twitter sends us data?
        here data will be a Python dict representing a tweet"""
        # only want to collect English-language tweets
        if data['lang'] == 'en':
            tweets.append(data)
            print("received tweet #", len(tweets))
        # stop when we've collected enough
        if len(tweets) >= 1000:
            self.disconnect()
def on_error(self, status_code, data):
    print(status_code, data)
    self.disconnect()

stream = MyStreamer(CONSUMER_KEY, CONSUMER_SECRET,
ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
# starts consuming public statuses that contain the keyword 'data'
stream.statuses.filter(track='data')
# if instead we wanted to start consuming a sample of *all* public statuses
# stream.statuses.sample()

from collections import Counter
top_hashtags = Counter(hashtag['text'].lower()
for tweet in tweets
for hashtag in tweet["entities"]["hashtags"])
print(top_hashtags.most_common(5))