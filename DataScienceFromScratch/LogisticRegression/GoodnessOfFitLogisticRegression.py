import TheLogisticFunction as logfn
import Vectors as vec
import ApplyingTheModel as apply
from matplotlib import pyplot as plt
true_positives = false_positives = true_negatives = false_negatives = 0

for x_i, y_i in zip(apply.x_test, apply.y_test):
    predict = logfn.logistic(vec.dot(apply.beta_hat, x_i))
    if y_i == 1 and predict >= 0.5: # TP: paid and we predict paid
        true_positives += 1
    elif y_i == 1: # FN: paid and we predict unpaid
        false_negatives += 1
    elif predict >= 0.5: # FP: unpaid and we predict paid
        false_positives += 1
    else: # TN: unpaid and we predict unpaid
        true_negatives += 1

precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)

print("-----------------------------------")
print(precision,recall)


predictions = [logfn.logistic(vec.dot(apply.beta_hat, x_i)) for x_i in apply.x_test]
plt.scatter(predictions, apply.y_test)
plt.xlabel("predicted probability")
plt.ylabel("actual outcome")
plt.title("Logistic Regression Predicted vs. Actual")
plt.show()