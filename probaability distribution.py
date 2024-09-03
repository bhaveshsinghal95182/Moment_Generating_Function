import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

class ProbabilityDistribution:
    def __init__(self, distribution):
        self.dist = distribution
        self.is_discrete = isinstance(self.dist, stats._distn_infrastructure.rv_discrete_frozen)

    def probability(self, x):
        """Calculate the probability density (for continuous) or mass (for discrete) function."""
        if self.is_discrete:
            return self.dist.pmf(x)
        else:
            return self.dist.pdf(x)

    def cumulative_probability(self, x):
        """Calculate the cumulative distribution function."""
        return self.dist.cdf(x)

    def expected_value(self):
        """Calculate the expected value (mean) of the distribution."""
        return self.dist.mean()

    def variance(self):
        """Calculate the variance of the distribution."""
        return self.dist.var()

    def moment_generating_function(self, t):
        """Calculate the moment generating function."""
        if isinstance(self.dist, (stats._distn_infrastructure.rv_frozen, stats._distn_infrastructure.rv_discrete_frozen)):
            if self.dist.dist.name == 'norm':
                # MGF for Normal distribution: exp(μt + (σ^2 * t^2) / 2)
                return np.exp(self.dist.mean() * t + (self.dist.var() * t**2) / 2)
            elif self.dist.dist.name == 'poisson':
                # MGF for Poisson distribution: exp(λ(e^t - 1))
                return np.exp(self.dist.mean() * (np.exp(t) - 1))
            elif self.dist.dist.name == 'expon':
                # MGF for Exponential distribution: λ / (λ - t), where λ is the rate parameter
                rate = 1 / self.dist.mean()
                return rate / (rate - t)
            else:
                raise NotImplementedError(f"MGF not implemented for {self.dist.dist.name} distribution")
        else:
            raise ValueError("Unsupported distribution type")

    def plot_distribution(self, start, end, num_points=1000):
        """Plot the probability density/mass function."""
        x = np.linspace(start, end, num_points)
        if self.is_discrete:
            x = np.arange(start, end + 1)
            y = self.probability(x)
            plt.figure(figsize=(10, 6))
            plt.bar(x, y, alpha=0.8, color='b')
            plt.title(f"Probability Mass Function - {self.dist.dist.name}")
            plt.xlabel("x")
            plt.ylabel("Probability Mass")
        else:
            y = self.probability(x)
            plt.figure(figsize=(10, 6))
            plt.plot(x, y)
            plt.title(f"Probability Density Function - {self.dist.dist.name}")
            plt.xlabel("x")
            plt.ylabel("Probability Density")
        plt.grid(True)
        plt.show()

# Example usage
if __name__ == "__main__":
    # Normal distribution
    normal_dist = ProbabilityDistribution(stats.norm(loc=0, scale=1))
    
    # Poisson distribution
    poisson_dist = ProbabilityDistribution(stats.poisson(mu=3))
    
    # Exponential distribution
    exponential_dist = ProbabilityDistribution(stats.expon(scale=1/2))  # rate = 2

    # Calculate and print some properties for the normal distribution
    print("Normal Distribution (μ=0, σ=1):")
    print(f"P(X <= 1) = {normal_dist.cumulative_probability(1):.4f}")
    print(f"E[X] = {normal_dist.expected_value():.4f}")
    print(f"Var(X) = {normal_dist.variance():.4f}")
    print(f"MGF at t=0.5: {normal_dist.moment_generating_function(0.5):.4f}")
    # Plot the PDF of the normal distribution
    normal_dist.plot_distribution(-4, 4)

    # Calculate and print some properties for the Poisson distribution
    print("\nPoisson Distribution (λ=3):")
    print(f"P(X = 2) = {poisson_dist.probability(2):.4f}")
    print(f"E[X] = {poisson_dist.expected_value():.4f}")
    print(f"Var(X) = {poisson_dist.variance():.4f}")
    print(f"MGF at t=0.5: {poisson_dist.moment_generating_function(0.5):.4f}")
    # Plot the PMF of the Poisson distribution
    poisson_dist.plot_distribution(0, 10)

    # Calculate and print some properties for the exponential distribution
    print("\nExponential Distribution (λ=2):")
    print(f"P(X <= 1) = {exponential_dist.cumulative_probability(1):.4f}")
    print(f"E[X] = {exponential_dist.expected_value():.4f}")
    print(f"Var(X) = {exponential_dist.variance():.4f}")
    print(f"MGF at t=0.5: {exponential_dist.moment_generating_function(0.5):.4f}")
    # Plot the PDF of the exponential distribution
    exponential_dist.plot_distribution(0, 3)