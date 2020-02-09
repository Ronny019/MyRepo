#                leukemia    no leukemia     total
#“Luke”           70          4,930           5,000
#not “Luke”      13,930      981,070         995,000
#total           14,000      986,000         1,000,000

def accuracy(tp, fp, fn, tn):
    correct = tp + tn
    total = tp + fp + fn + tn
    return correct / total

print(accuracy(70, 4930, 13930, 981070)) # 0.98114

def precision(tp, fp, fn, tn):
    return tp / (tp + fp)

print(precision(70, 4930, 13930, 981070)) # 0.014

def recall(tp, fp, fn, tn):
    return tp / (tp + fn)

print(recall(70, 4930, 13930, 981070)) # 0.005


def f1_score(tp, fp, fn, tn):
    p = precision(tp, fp, fn, tn)
    r = recall(tp, fp, fn, tn)
    return 2 * p * r / (p + r)
