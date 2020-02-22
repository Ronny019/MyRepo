import Digression_TheBootstrap as boot
import FittingTheModel as fit
import NormalDistribution as nor
import DescribingASingleSetOfData as des
def estimate_sample_beta(sample):
    """sample is a list of pairs (x_i, y_i)"""
    x_sample, y_sample = zip(*sample) # magic unzipping trick
    return fit.estimate_beta(x_sample, y_sample)

bootstrap_betas = boot.bootstrap_statistic(list(zip(fit.x, fit.daily_minutes_good)),
                                        estimate_sample_beta,
                                        100)
print("------------------------------------------------------------------")
for beta in bootstrap_betas:
    print(beta)


bootstrap_standard_errors = [des.standard_deviation([beta[i] for beta in bootstrap_betas])
                             for i in range(4)]

print("Bootstrap standard errors:")
print(bootstrap_standard_errors)



def p_value(beta_hat_j, sigma_hat_j):
    if beta_hat_j > 0:
        # if the coefficient is positive, we need to compute twice the
        # probability of seeing an even *larger* value
        return 2 * (1 - nor.normal_cdf(beta_hat_j / sigma_hat_j))
    else:
        # otherwise twice the probability of seeing a *smaller* value
        return 2 * nor.normal_cdf(beta_hat_j / sigma_hat_j)


print(p_value(30.63, 1.174)) # ~0 (constant term)
print(p_value(0.972, 0.079)) # ~0 (num_friends)
print(p_value(-1.868, 0.131)) # ~0 (work_hours)
print(p_value(0.911, 0.990)) # 0.36 (phd)