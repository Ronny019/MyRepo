import CreatingADecisionTree as create
from functools import partial
hiring_tree = ('level',
                {'Junior': ('phd', {'no': True, 'yes': False}),
                'Mid': True,
                'Senior': ('tweets', {'no': False, 'yes': True})})


def classify(tree, input):
    """classify the input using the given decision tree"""
    # if this is a leaf node, return its value
    if tree in [True, False]:
        return tree
    # otherwise this tree consists of an attribute to split on
    # and a dictionary whose keys are values of that attribute
    # and whose values of are subtrees to consider next
    attribute, subtree_dict = tree
    subtree_key = input.get(attribute) # None if input is missing attribute
    if subtree_key not in subtree_dict: # if no subtree for key,
        subtree_key = None # we'll use the None subtree
    subtree = subtree_dict[subtree_key] # choose the appropriate subtree
    return classify(subtree, input) # and use it to classify the input




def build_tree_id3(inputs, split_candidates=None):# if this is our first pass,
    # all keys of the first input are split candidates
    if split_candidates is None:
        split_candidates = inputs[0][0].keys()
    # count Trues and Falses in the inputs
    num_inputs = len(inputs)
    num_trues = len([label for item, label in inputs if label])
    num_falses = num_inputs - num_trues
    if num_trues == 0: return False # no Trues? return a "False" leaf
    if num_falses == 0: return True # no Falses? return a "True" leaf
    if not split_candidates: # if no split candidates left
        return num_trues >= num_falses # return the majority leaf
    # otherwise, split on the best attribute
    best_attribute = min(split_candidates,
                        key=partial(create.partition_entropy_by, inputs))
    partitions = create.partition_by(inputs, best_attribute)
    new_candidates = [a for a in split_candidates
                    if a != best_attribute]
    # recursively build the subtrees
    subtrees = { attribute_value : build_tree_id3(subset, new_candidates)
                for attribute_value, subset in partitions.items() }
    subtrees[None] = num_trues > num_falses # default case
    return (best_attribute, subtrees)



tree = build_tree_id3(create.inputs)
print(classify(tree, { "level" : "Junior",
                "lang" : "Java",
                "tweets" : "yes",
                "phd" : "no"} )) # True
print(classify(tree, { "level" : "Junior",
                "lang" : "Java",
                "tweets" : "yes",
                "phd" : "yes"} )) # False

print(classify(tree, { "level" : "Intern" } )) # True
print(classify(tree, { "level" : "Senior" } )) # False