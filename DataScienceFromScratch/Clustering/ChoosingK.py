import TheModelClustering as mod
import Example_Meetups as ex
import Vectors as vec
from matplotlib import pyplot as plt

def squared_clustering_errors(inputs, k):
    """finds the total squared error from k-means clustering the inputs"""
    clusterer = mod.KMeans(k)
    clusterer.train(ex.inputs)
    means = clusterer.means
    assignments = list(map(clusterer.classify, inputs))
    return sum(vec.squared_distance(input, means[cluster])
                for input, cluster in zip(ex.inputs, assignments))


ks = range(1, len(ex.inputs) + 1)
errors = [squared_clustering_errors(ex.inputs, k) for k in ks]
plt.plot(ks, errors)
plt.xticks(ks)
plt.xlabel("k")
plt.ylabel("total squared error")
plt.title("Total Error vs. # of Clusters")
plt.show()