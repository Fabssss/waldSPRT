import math
import time

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as com

class WaldSPRT:
    def __init__(self, alpha, beta, epsilon, patience, min_alpha):
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.patience = patience
        self.min_alpha = min_alpha
        self.prob_list = []

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

    def estimate_probability(self, data, p0, p1):
        while abs(p1 - p0) > self.epsilon:
            p_new = (p0 + p1) / 2

            result = self.walds_sprt(data, p0, p_new)

            if result == "H1":
                p0 = p_new
            else:
                p1 = p_new

        return (p0 + p1) / 2

    def early_stopping(self):
        if len(self.prob_list) < self.patience:
            return 1

        checklist = self.prob_list[-self.patience:]
        change = max(checklist) - min(checklist)

        if change < self.min_alpha:
            return 0

        return 1

##settings
guess_prob = 0.0
guess_tolerance = 0
alpha = 1e-10
beta = 1e-10
epsilon = 1e-30
shrink_factor = 0.5

guess_prob_list = []
sample_list = []
actual_prob = 0.0
population_size = 0
st.title("Welcome to HEADS/TAILS Predictor")
# gif embed
com.html("<div style='display: flex; align-items: center; justify-content: center;'><img src='https://i.gifer.com/Fw3P.gif' width='100' height='100'></div>")

form = st.form(key='input_form')
heads = int(form.number_input(label='Enter No. Of HEADS in Population'))
tails = int(form.number_input(label='Enter No. Of TAILS in Population'))
sampling_rate = int(form.slider(label = 'Select Sample Size',min_value= 100, max_value= 1000,step= 100))
submit_button = form.form_submit_button(label='Submit')
if submit_button:
    population_size = heads+tails
    actual_prob = heads/population_size

population_data = np.random.binomial(1, actual_prob, population_size)
sample_size = sampling_rate

sample = st.empty()
prob = st.empty()
sprt = WaldSPRT(alpha, beta, epsilon, 2000, 1)
iteration_list = [0]
p0 = 0.0
p1 = 1.0
while abs(actual_prob - guess_prob) > guess_tolerance and sample_size < population_size:

    subsample = population_data[0:sample_size]
    guess_prob = sprt.estimate_probability(subsample, p0, p1)
    print(f'Sample Size:{sample_size}')
    sample.header(f'SAMPLE SIZE: {sample_size}')
    sample_list.append(sample_size)
    print(f'Guess:{guess_prob}')
    prob.header(f'ESTIMATED PROB OF HEAD: {guess_prob}')
    guess_prob_list.append(guess_prob)
    sprt.prob_list.append(guess_prob)
    iteration_list.append(iteration_list[-1] + 1)

    if sprt.early_stopping() != 1:
        st.markdown("EARLY STOPPING")
        break
    p0 = max(p0, guess_prob - (p1 - p0) * shrink_factor)  # Update p0 with a reduced range
    p1 = min(p1, guess_prob + (p1 - p0) * shrink_factor)  # Update p1 with a reduced range

    sample_size = sample_size + sampling_rate
    time.sleep(1)

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('Predicted Probability')
ax.set_ylabel('Sample Size')
ax.set_zlabel("Iteration")
ax.scatter(guess_prob_list, sample_list, iteration_list[:-1])

# Display the plot in Streamlit
st.pyplot(fig)

