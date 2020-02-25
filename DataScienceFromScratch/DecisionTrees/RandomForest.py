from collections import Counter

def forest_classify(trees, input):
    votes = [classify(tree, input) for tree in trees]
    vote_counts = Counter(votes)
    return vote_counts.most_common(1)[0][0]



## if there's already few enough split candidates, look at all of them
#if len(split_candidates) <= self.num_split_candidates:
#sampled_split_candidates = split_candidates
## otherwise pick a random sample
#else:
#sampled_split_candidates = random.sample(split_candidates,
#self.num_split_candidates)
## now choose the best attribute only from those candidates
#best_attribute = min(sampled_split_candidates,
#key=partial(partition_entropy_by, inputs))
#partitions = partition_by(inputs, best_attribute)