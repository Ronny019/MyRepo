#Have the Spam Assassin Corpus under the path C:\Spam Assassin Corpus
import glob, re
import Implementation as imp
import random
import OverfittingAndUnderfitting as over_und
from collections import Counter
# modify the path with wherever you've put the files
path = r"C:\Spam Assassin Corpus\*\*"
data = []
# glob.glob returns every filename that matches the wildcarded path
for fn in glob.glob(path):
    is_spam = "ham" not in fn
    with open(fn,'r',errors="ignore") as file:
        for line in file:
            if line.startswith("Subject:"):
                # remove the leading "Subject: " and keep what's left
                subject = re.sub(r"^Subject: ", "", line).strip()
                data.append((subject, is_spam))

print(len(data))


random.seed(0) # just so you get the same answers as me
train_data, test_data = over_und.split_data(data, 0.75)
classifier = imp.NaiveBayesClassifier()
classifier.train(train_data)


# triplets (subject, actual is_spam, predicted spam probability)
classified = [(subject, is_spam, classifier.classify(subject))
                for subject, is_spam in test_data]

# assume that spam_probability > 0.5 corresponds to spam prediction
# and count the combinations of (actual is_spam, predicted is_spam)
counts = Counter((is_spam, spam_probability > 0.5)
for _, is_spam, spam_probability in classified)

print(counts)