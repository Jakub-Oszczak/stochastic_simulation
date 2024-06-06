import numpy as np
import matplotlib.pyplot as plt

# Step 1: Simulate Geometric Distribution
def simulate_geometric_distribution(probability_values, n_samples):
    results = {}
    for p in probability_values:
        results[p] = np.random.geometric(p, n_samples)
    return results

# Step 2: Simulate 6-point Distribution
# (a) Direct method
def direct_method(points, probabilities, n_samples):
    return np.random.choice(points, size=n_samples, p=probabilities)

# (b) Rejection method
def rejection_method(points, probabilities, n_samples):
    proposal_prob = np.ones(len(points)) / len(points)
    M = max(probabilities / proposal_prob)
    samples = []
    while len(samples) < n_samples:
        index = np.random.choice(range(len(points)), p=proposal_prob)
        if np.random.rand() < probabilities[index] / proposal_prob[index]:
            samples.append(points[index])
    return samples

# (c) Alias method
class AliasMethod:
    def __init__(self, probabilities):
        self.n = len(probabilities)
        self.prob = np.zeros(self.n)
        self.alias = np.zeros(self.n, dtype=np.int32)
        scaled_probabilities = np.array(probabilities) * self.n
        small, large = [], []
        for i, prob in enumerate(scaled_probabilities):
            if prob < 1.0:
                small.append(i)
            else:
                large.append(i)
        while small and large:
            s = small.pop()
            l = large.pop()
            self.prob[s] = scaled_probabilities[s]
            self.alias[s] = l
            scaled_probabilities[l] += scaled_probabilities[s] - 1
            if scaled_probabilities[l] < 1.0:
                small.append(l)
            else:
                large.append(l)
        while large:
            l = large.pop()
            self.prob[l] = 1
        while small:
            s = small.pop()
            self.prob[s] = 1
    def generate(self):
        index = np.random.randint(0, self.n)
        return index + 1 if np.random.rand() < self.prob[index] else self.alias[index] + 1

# Defining parameters and probabilities
p_values = [0.05, 0.2, 0.5]
points = np.array([1, 2, 3, 4, 5, 6])
probabilities = np.array([7/48, 5/48, 1/8, 1/16, 1/4, 5/16])
samples = 10000

# Generate data
geometric_data = simulate_geometric_distribution(p_values, samples)
direct_samples = direct_method(points, probabilities, samples)
rejection_samples = rejection_method(points, probabilities, samples)
alias_generator = AliasMethod(probabilities)
alias_samples = [alias_generator.generate() for _ in range(samples)]

# Visualization: Histogram for geometric distribution
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, p in zip(axes, p_values):
    ax.hist(geometric_data[p], bins=max(geometric_data[p]), density=True, alpha=0.6)
    ax.set_title(f'Geometric distribution, p = {p}')
    ax.set_xlabel('Number of Trials')
    ax.set_ylabel('Probability')
plt.tight_layout()

# Visualization: Comparing methods for 6-point distribution
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].hist(direct_samples, bins=np.arange(0.5, 7.5, 1), rwidth=0.8, density=True, alpha=0.75)
axes[0].set_title('Direct Method')
axes[1].hist(rejection_samples, bins=np.arange(0.5, 7.5, 1), rwidth=0.8, density=True, alpha=0.75)
axes[1].set_title('Rejection Method')
axes[2].hist(alias_samples, bins=np.arange(0.5, 7.5, 1), rwidth=0.8, density=True, alpha=0.75)
axes[2].set_title('Alias Method')
for ax in axes:
    ax.set_xlabel('Point')
    ax.set_ylabel('Probability')
    ax.set_xticks(points)
plt.tight_layout()
plt.show()
