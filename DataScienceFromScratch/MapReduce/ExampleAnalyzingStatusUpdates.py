import MapReduceMoreGenerally as map_red
import datetime
from collections import Counter
import Implementation as imp
status_updates = [
                    {"id": 1,
                     "username" : "joelgrus",
                     "text" : "Is anyone interested in a data science book? lol lol lol lol",
                     "created_at" : datetime.datetime(2013, 12, 21, 11, 47, 0),
                     "liked_by" : ["data_guy", "data_gal", "bill"] },
                        # add your own
                 ]


def data_science_day_mapper(status_update):
    """yields (day_of_week, 1) if status_update contains "data science" """
    if "data science" in status_update["text"].lower():
        day_of_week = status_update["created_at"].weekday()
        yield (day_of_week, 1)



data_science_days = map_red.map_reduce(status_updates,
                                       data_science_day_mapper,
                                       map_red.sum_reducer)

print(data_science_days)


def words_per_user_mapper(status_update):
    user = status_update["username"]
    for word in (status_update["text"]).split():
        yield (user, (word, 1))



def most_popular_word_reducer(user, words_and_counts):
    """given a sequence of (word, count) pairs,
    return the word with the highest total count"""
    word_counts = Counter()
    for word, count in words_and_counts:
        word_counts[word] += count
    word, count = word_counts.most_common(1)[0]
    yield (user, (word, count))


user_words = map_red.map_reduce(status_updates,
                        words_per_user_mapper,
                        most_popular_word_reducer)

print(user_words)


def liker_mapper(status_update):
    user = status_update["username"]
    for liker in status_update["liked_by"]:
        yield (user, liker)


distinct_likers_per_user = map_red.map_reduce(status_updates,
                                        liker_mapper,
                                        map_red.count_distinct_reducer)

print(distinct_likers_per_user)
