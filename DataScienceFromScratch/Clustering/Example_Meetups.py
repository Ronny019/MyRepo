import TheModelClustering as mod
import random
inputs = [[-14,-5],[13,13],[20,23],[-19,-11],[-9,-16],[21,27],[-49,15],[26,13],[-46,5],[-34,-1],[11,15],[-49,0],[-22,-16],[19,28],[-12,-8],[-13,-19],[-41,8],[-11,-6],[-25,-9],[-18,-3]]

random.seed(0) # so you get the same results as me
clusterer = mod.KMeans(3)
clusterer.train(inputs)
print("3-means:")
print(clusterer.means)
print()


random.seed(0)
clusterer = mod.KMeans(2)
clusterer.train(inputs)
print("2-means:")
print(clusterer.means)
print()