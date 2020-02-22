import random
import DescribingASingleSetOfData as des
def bootstrap_sample(data):
    """randomly samples len(data) elements with replacement"""
    return [random.choice(data) for _ in data]

def bootstrap_statistic(data, stats_fn, num_samples):
    """evaluates stats_fn on num_samples bootstrap samples from data"""
    return [stats_fn(bootstrap_sample(data))
            for _ in range(num_samples)]


# 101 points all very close to 100
close_to_100 = [99.5 + random.random() for _ in range(101)]


# 101 points, 50 of them near 0, 50 of them near 200
far_from_100 = ([99.5 + random.random()] +
                [random.random() for _ in range(50)] +
                [200 + random.random() for _ in range(50)])

bs_close = bootstrap_statistic(close_to_100, des.median, 100)

bs_far = bootstrap_statistic(far_from_100, des.median, 100)

print(bs_close)

print(bs_far)
