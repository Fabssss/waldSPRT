import math
import os
import numpy as np
import matplotlib.pyplot as plt
import imageio


class WaldSPRT:

    def __init__(self, alpha, beta, epsilon):
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon

    @staticmethod
    def log_likelihood(y, n, p):
        log_comb = math.lgamma(n + 1) - math.lgamma(y + 1) - math.lgamma(n - y + 1)

        if p == 0:
            log_prob_y = 0 if y == 0 else float('-inf')
            log_prob_n_y = math.log(1 - p) * (n - y)
        elif p == 1:
            log_prob_y = math.log(p) * y
            log_prob_n_y = 0 if n - y == 0 else float('-inf')
        else:
            log_prob_y = math.log(p) * y
            log_prob_n_y = math.log(1 - p) * (n - y)

        return log_comb + log_prob_y + log_prob_n_y

    def walds_sprt(self, data, p0, p1):
        A = math.log(self.beta / (1 - self.alpha))
        B = math.log((1 - self.beta) / self.alpha)

        n = len(data)
        y = np.count_nonzero(data)

        log_likelihood_h0 = self.log_likelihood(y, n, p0)
        log_likelihood_h1 = self.log_likelihood(y, n, p1)

        LR = log_likelihood_h1 - log_likelihood_h0

        if LR <= A:
            return "H0"
        elif LR >= B:
            return "H1"
        else:
            return "Continue collecting data"

    def plot_walds_sprt(self, data, p0, p1, iteration):
        A = math.log(self.beta / (1 - self.alpha))
        B = math.log((1 - self.beta) / self.alpha)

        n = len(data)
        y = np.count_nonzero(data)

        p_values = np.linspace(0, 1, 1000)
        likelihood_ratios = [self.log_likelihood(y, n, p) - self.log_likelihood(y, n, p0) if p > 0 and p < 1 else float('-inf') for p in p_values]

        plt.plot(p_values, likelihood_ratios, label='Likelihood Ratio')
        plt.axhline(y=A, color='r', linestyle='--', label='Decision Boundary A')
        plt.axhline(y=B, color='g', linestyle='--', label='Decision Boundary B')
        plt.xlabel('Population Probability')
        plt.ylabel('Log Likelihood Ratio')
        plt.legend()

        p_new = (p0 + p1) / 2
        current_lr = self.log_likelihood(y, n, p_new) - self.log_likelihood(y, n, p0)
        margin = (B - A) / 2

        if not math.isinf(current_lr):
            plt.ylim(current_lr - margin, current_lr + margin)
        else:
            fallback_margin = 0.1
            plt.ylim(B - fallback_margin, A + fallback_margin)

        x_margin = 0.1
        plt.xlim(p0 - x_margin, p1 + x_margin)

        output_dir = 'plots'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(f"{output_dir}/iteration_{iteration}.png")
        plt.close()

    def estimate_probability(self, data, p0, p1):
        iteration = 1
        while abs(p1 - p0) > self.epsilon:
            p_new = (p0 + p1) / 2

            result = self.walds_sprt(data, p0, p_new)

            self.plot_walds_sprt(data, p0, p_new, iteration)

            if result == "H1":
                p0 = p_new
            else:
                p1 = p_new

            iteration += 1

        return (p0 + p1) / 2


def create_gif(input_folder, output_filename, duration=0.5):
    filenames =sorted(os.listdir(input_folder))
    images = [imageio.imread(os.path.join(input_folder, filename)) for filename in filenames if filename.endswith('.png')]
    imageio.mimsave(output_filename, images, duration=duration)


# Example usage
sample_data = np.random.binomial(1, 0.7, 200000)
p0 = 0.0
p1 = 1.0
alpha = 1e-5
beta = 1e-5
epsilon = 1e-20

sprt = WaldSPRT(alpha, beta, epsilon)
estimated_probability = sprt.estimate_probability(sample_data, p0, p1)
print(estimated_probability)

input_folder = 'plots'
output_filename = 'wald_sprt.gif'
duration = 0.2  # Adjust this value to control the duration of each frame in the GIF

create_gif(input_folder, output_filename, duration)