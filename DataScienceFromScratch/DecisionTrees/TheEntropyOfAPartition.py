import Entropy as ent

def partition_entropy(subsets):
    """find the entropy from this partition of data into subsets
    subsets is a list of lists of labeled data"""
    total_count = sum(len(subset) for subset in subsets)
    return sum( ent.data_entropy(subset) * len(subset) / total_count
                for subset in subsets )