import Vectors as vec
import random
import statistics
from matplotlib import pyplot as plt
def random_point(dim):
    return [random.random() for _ in range(dim)]

def random_distances(dim, num_pairs):
    return [vec.distance(random_point(dim), random_point(dim))
            for _ in range(num_pairs)]

dimensions = range(1, 101)
avg_distances = []
min_distances = []
random.seed(0)
for dim in dimensions:
    distances = random_distances(dim, 10000) # 10,000 random pairs
    means = statistics.mean(distances)
    avg_distances.append(means) # track the average
    mins = min(distances)
    min_distances.append(mins) # track the minimum

min_avg_ratio = [min_dist / avg_dist
                for min_dist, avg_dist in zip(min_distances, avg_distances)]