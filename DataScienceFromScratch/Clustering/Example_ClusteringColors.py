#have the image files in the location C:\test_images

import matplotlib.image as mpimg
import TheModelClustering as  mod
from matplotlib import pyplot as plt

path_to_png_file = r"C:\test_images\img8.png" # wherever your image is
img = mpimg.imread(path_to_png_file)


#top_row = img[0]
#top_left_pixel = top_row[0]
#red, green, blue = top_left_pixel

#print(red, green, blue)

pixels = list([list(pixel) for row in img for pixel in row])

clusterer = mod.KMeans(5)
clusterer.train(pixels)


def recolor(pixel):
    cluster = clusterer.classify(pixel) # index of the closest cluster
    return clusterer.means[cluster] # mean of the closest cluster

new_img = [[recolor(pixel) for pixel in row] # recolor this row of pixels
                  for row in img]

plt.imshow(new_img)
plt.axis('off')
plt.show()