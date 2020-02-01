import DescribingASingleSetOfData as dat
import Vectors as vec
def covariance(x, y):
    n = len(x)
    return vec.dot(dat.de_mean(x), dat.de_mean(y)) / (n - 1)

#covariance(dat.num_friends, daily_minutes) # 22.43

def correlation(x, y):
    stdev_x = dat.standard_deviation(x)
    stdev_y = dat.standard_deviation(y)
    if stdev_x > 0 and stdev_y > 0:
        return covariance(x, y) / stdev_x / stdev_y
    else:
        return 0 # if no variation, correlation is zero

#correlation(num_friends, daily_minutes) # 0.25

#Code for removing outlier and then calculating correlation. Here we remove the outlier 100 who only spends 1 min per day on DataSciencester
#outlier = dat.num_friends.index(100) # index of outlier
#num_friends_good = [x
#for i, x in enumerate(num_friends)
#if i != outlier]
#daily_minutes_good = [x
#for i, x in enumerate(daily_minutes)
#if i != outlier]
#correlation(num_friends_good, daily_minutes_good) # 0.57