import RecommenderSystems as rec
from collections import Counter

popular_interests = Counter(interest
                            for user_interests in rec.users_interests
                            for interest in user_interests).most_common()

print(popular_interests)


def most_popular_new_interests(user_interests, max_results=5):
    suggestions = [(interest, frequency)
                    for interest, frequency in popular_interests
                    if interest not in user_interests]
    return suggestions[:max_results]


print()
print(most_popular_new_interests(rec.users_interests[1], 5))

print(most_popular_new_interests(rec.users_interests[3], 5))