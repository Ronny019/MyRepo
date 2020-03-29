import UserBasedCollaborativeFiltering as user
import RecommenderSystems as rec
from collections import defaultdict
interest_user_matrix = [[user_interest_vector[j]
                        for user_interest_vector in user.user_interest_matrix]
                        for j, _ in enumerate(user.unique_interests)]

print("-------------------------------------------")
print(interest_user_matrix)

interest_similarities = [[user.cosine_similarity(user_vector_i, user_vector_j)
                          for user_vector_j in interest_user_matrix]
                          for user_vector_i in interest_user_matrix]


def most_similar_interests_to(interest_id):
    similarities = interest_similarities[interest_id]
    pairs = [(user.unique_interests[other_interest_id], similarity)
            for other_interest_id, similarity in enumerate(similarities)
            if interest_id != other_interest_id and similarity > 0]
    return sorted(pairs,
                key=lambda similarities: similarities[1],
                reverse=True)

print(most_similar_interests_to(0))


def item_based_suggestions(user_id, include_current_interests=False):
    # add up the similar interests
    suggestions = defaultdict(float)
    user_interest_vector = user.user_interest_matrix[user_id]
    for interest_id, is_interested in enumerate(user_interest_vector):
        if is_interested == 1:
            similar_interests = most_similar_interests_to(interest_id)
            for interest, similarity in similar_interests:
                suggestions[interest] += similarity
    # sort them by weight
    suggestions = sorted(suggestions.items(),
                        key=lambda similarities: similarities[1],
                        reverse=True)
    if include_current_interests:
        return suggestions
    else:
        return [(suggestion, weight)
                for suggestion, weight in suggestions
                if suggestion not in rec.users_interests[user_id]]

print()
print(item_based_suggestions(0))