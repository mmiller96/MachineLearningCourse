import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.utils import check_random_state
from scipy.stats import norm

class BayesianOptimizer:
    """ Bayesian Optimizer, this class is already complete! """

    def __init__(self, acquisition_function, initial_random_samples=3,
                 random_state=0):
        """ The constructor, initializes the optimizer

        Parameters
        ----------
        acquisition_function: callable
            The acquisition function for the optimization
        initial_random_samples:
            How many query points should be samples randomly, before the
            GPR model is trained and used.
        random_state:
            Seed for the random number generator
        """

        self.acquisition_function = acquisition_function
        self.initial_random_samples = initial_random_samples

        self.rng = check_random_state(random_state)

        self.X = []
        self.y = []

        kernel = RBF(length_scale=0.1, length_scale_bounds=(0.01, 1.0))
        self.gpr_model = GaussianProcessRegressor(kernel=kernel)
        self.x0 = np.asarray([0])

    def get_next_query_point(self, bounds):
        """ Suggest a new query point to evaluate the objective function at

        Parameters
        ----------
        bounds: tuple, shape(2)
            lower and upper bound for the query point to suggest
        """
        if len(self.X) < self.initial_random_samples:
            x_query = np.asarray([self.rng.uniform(low=bounds[0],
                                                   high=bounds[1])])
        else:                                                                   # we have changed the program, because we couldn't fix the problem with the scipy optimization
            x = np.linspace(-2, 2, 1000).reshape(-1, 1)
            acc_val = self.acquisition_function(x, self.gpr_model)              # calculate in a grid the acquisition function and take the position of the maximum
            ix = np.argmax(acc_val)
            x_query = x[ix]
        return x_query

    def update_model(self, X, y):
        """ Update the Gaussian Process Regeression Model based on the returned
            value of the objective function.

        Parameters
        ----------
        X: ndarray, shape [n_samples, n_dims]
            The point(s) the objective function had been evaluated
        y: ndarraym shape [n_samples,]
            Corresponding values of the objetive function
        """
        self.X.append(X)
        self.y.append(y)

        if len(self.X) >= self.initial_random_samples:
            self.gpr_model.fit(self.X, self.y)

class UpperConfidenceBound:
    """ The Upper Confidence Bound Acquistion Function. The class is defined as
        so called Functor class, making each object created from it executable
        (like a regular function). """

    def __init__(self, kappa=0.1):  # kappa --> add more exploration
        self.kappa = kappa
        """ The constructor, taking the hyperparameters for the
            acquisition function.

        Parameters
        ----------
            ....
        """
        # YOUR IMPLEMENTATION


    def __call__(self, x, model):   # equation for upper confidence bound, see exercise for a more detailed description
        """__call__: This method makes any object from the class callable like
                           a regular function.

        Parameters
        ----------
        x: ndarray, shape [n_dims,]
            The point to evaluate the acquisition function at
        model: sklearn.gaussian_process.GaussianProcessRegressor
            The gaussian process regression model for the objective function
        """
        mu, std = model.predict(x.reshape(-1,1), return_std=True)
        ucb = mu.ravel() + self.kappa * std.ravel()
        return ucb


        # YOUR IMPLEMENTATION
class ExpectedImprovement:
    def __init__(self, xi=0.01):
        self.xi = xi
    def __call__(self, x, model):   # equation for expected improvement, see exercise for a more detailed description
        self.model = model
        mu, sigma = model.predict(x.reshape(-1,1), return_std=True)
        mu_sample = model.predict(model.X_train_)

        sigma = sigma.reshape(-1, 1)
        mu_sample_opt = np.max(mu_sample)

        imp = mu - mu_sample_opt - self.xi
        Z = imp / (sigma+1E-9)          # we have added a little term to sigma to avoid division with zero
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        return ei

class ProbabilityImprovement:
    def __init__(self, xi=0.01):    # xi --> penalize greed
        self.xi = xi
    def __call__(self, x, model): # equation for probability improvement, see exercise for a more detailed description
        self.model = model
        mu, sigma = model.predict(x.reshape(-1, 1), return_std=True)
        mu_sample = model.predict(model.X_train_)

        sigma = sigma.reshape(-1, 1)
        mu_sample_opt = np.max(mu_sample)

        pi = norm.cdf((mu-mu_sample_opt-self.xi)/(sigma+1E-9))  # we have added a little term to sigma to avoid division with zero
        return pi

def f(x):
    """ The objective function to optimize. Note: Usally this function is not
        available in analytic form, when bayesian optimization is applied. Note
                 that it is noisy.

    Parameters
    ----------
    x:
        The point(s) to evaluate the objective function at.
    """
    return (1.0 - np.tanh(x**2)) * np.sin(x*6) + np.random.normal(loc=0.0,
                                                                  scale=0.02)
def plot_gaussian_process(bo,bounds, save_figure=False, i=0, name=r'ExpectedImprovement', path_folder=r'C:\Users\Data Miner\Desktop\Uni\MachineLearning\ex04\\'):
    x = np.linspace(bounds[0], bounds[1], 1000).reshape(-1,1)
    y_acq = bo.acquisition_function(x, bo.gpr_model)
    x_new = x[np.argmax(y_acq)]
    y_mean, y_sigma = bo.gpr_model.predict(x, return_std=True)
    y_low = y_mean - 1.96 * y_sigma.reshape(-1, 1)              # 1.96 std --> 95% confidence
    y_high = y_mean + 1.96 * y_sigma.reshape(-1, 1)

    plt.subplot(1, 2, 1)
    plt.fill_between(x.ravel(), y_low.ravel(), y_high.ravel())
    plt.scatter(bo.X, bo.y)
    plt.plot(x, f(x), color='y')
    plt.axvline(x=x_new, ls='--', c='k', lw=1, label='Next sampling location')
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(x, y_acq, 'r-', lw=1, label='Acquisition function')
    plt.axvline(x=x_new, ls='--', c='k', lw=1, label='Next sampling location')
    if(save_figure): plt.savefig(path_folder+name+str(i)+'.pdf')
    plt.show()

def plot_acquisition(X, Y, X_next, show_legend=False):
    plt.plot(X, Y, 'r-', lw=1, label='Acquisition function')
    plt.axvline(x=X_next, ls='--', c='k', lw=1, label='Next sampling location')


if __name__ == "__main__":
    bounds = [-2, 2]
    if(False):  # plot the graphs for a specific acquisition function
        #acquisition = ExpectedImprovement(xi=0.01)          # activate just one acquisition function
        #acquisition = UpperConfidenceBound(kappa=0.2)
        acquisition = ProbabilityImprovement(xi=0.01)
        bo = BayesianOptimizer(acquisition, random_state=42)
        for i in range(15):
            x_new = bo.get_next_query_point(bounds)     # get the next best x depending on the acqusition function
            bo.update_model(x_new, f(x_new))            # update the gaussian process model
            if(i>=3): plot_gaussian_process(bo, bounds, save_figure=False, i=i, name='ExpectedImprovement')     # plot gaussian process and the acquisition function. With the optional parameters you can save the figure
    else:   # calculate the mean and std of multiple runs with different acquisition functions.                 # start with 3 after randomization
        acquisitions = [ExpectedImprovement(xi=0.01), UpperConfidenceBound(kappa=0.2), ProbabilityImprovement(xi=0.01)]
        best_acq = []
        for acquisition in acquisitions:
            best = []
            for j in range(10):
                bo = BayesianOptimizer(acquisition)
                for i in range(10):
                    x_new = bo.get_next_query_point(bounds)
                    x = np.linspace(bounds[0], bounds[1], 1000).reshape(-1, 1)
                    bo.update_model(x_new, f(x_new))
                best.append(np.max(bo.y))
            best_acq.append(np.array(best))
        print('ExpectedImprovement: mean {0}, std {1}'.format(best_acq[0].mean(), best_acq[0].std()))
        print('UpperConfidenceBound: mean {0}, std {1}'.format(best_acq[1].mean(), best_acq[1].std()))
        print('ProbabilityImprovement: mean {0}, std {1}'.format(best_acq[2].mean(), best_acq[2].std()))



