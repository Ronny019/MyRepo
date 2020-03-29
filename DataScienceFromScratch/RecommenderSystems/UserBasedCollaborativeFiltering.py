import math
import Vectors as vec
import RecommenderSystems as rec
from collections import defaultdict

def cosine_similarity(v, w):
    return vec.dot(v, w) / math.sqrt(vec.dot(v, v) * vec.dot(w, w))


unique_interests = sorted(list({ interest
                                for user_interests in rec.users_interests
                                for interest in user_interests }))

print(unique_interests)


def make_user_interest_vector(user_interests):
    """given a list of interests, produce a vector whose ith element is 1
    if unique_interests[i] is in the list, 0 otherwise"""
    return [1 if interest in user_interests else 0
            for interest in unique_interests]


user_interest_matrix = list(map(make_user_interest_vector, rec.users_interests))

print()

user_similarities = [[cosine_similarity(interest_vector_i, interest_vector_j)
                      for interest_vector_j in user_interest_matrix]
                      for interest_vector_i in user_interest_matrix]

for user_similarity in user_similarities:
    print(user_similarity)


def most_similar_users_to(user_id):
    pairs = [(other_user_id, similarity) # find other
            for other_user_id, similarity in # users with
            enumerate(user_similarities[user_id]) # nonzero
                    if user_id != other_user_id and similarity > 0] # similarity
    return sorted(pairs, # sort them
    key=lambda id_sim: id_sim[1], # most similar
    reverse=True)

print(most_similar_users_to(0))


def user_based_suggestions(user_id, include_current_interests=False):
    # sum up the similarities
    suggestions = defaultdict(float)
    for other_user_id, similarity in most_similar_users_to(user_id):
        for interest in rec.users_interests[other_user_id]:
            suggestions[interest] += similarity

    # convert them to a sorted list
    suggestions = sorted(suggestions.items(),
                        key=lambda suggestions: suggestions[1],
                        reverse=True)

    # and (maybe) exclude already-interests
    if include_current_interests:
        return suggestions
    else:
        return [(suggestion, weight)
                for suggestion, weight in suggestions
                if suggestion not in rec.users_interests[user_id]]

print(user_based_suggestions(0))