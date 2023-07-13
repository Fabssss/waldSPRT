# waldSPRT

## Overview
This repository demonstrates the usage of sequential data analysis to predict the binomial population distribution from a small sample. The research combines the techniques of Wald's Sequential Probability Ratio Test (SPRT) and a Binary Search approach to search for the most optimal probability for each sample. The work is inspired by Dr. Manoranjan Sathpathy et al.'s analysis and prediction of Covid-19 cases using the Wald SPRT test in the paper titled "Effect of Migrant Labourer Inflow on the Early Spread of Covid-19 in Odisha: A Case Study".

## Description
The main objective of this project is to estimate the probability of a binomial population distribution based on a small sample using sequential analysis. The Wald SPRT algorithm is implemented to iteratively update the probability estimate until a desired level of accuracy is achieved.

## Features
- Estimation of binomial population distribution probability
- Sequential data analysis using Wald's Sequential Probability Ratio Test
- Binary Search approach for optimal probability search
- Integration with Streamlit for interactive demonstration

## Dependencies
- Python 3.x
- numpy
- matplotlib
- streamlit

## Usage
1. Install the required dependencies using `pip install -r requirements.txt`.
2. Run the Streamlit app by executing `streamlit run test.py`.
3. Access the application through the provided URL in the terminal.

## Additional Notes
- The data used for demonstration can be customized in the `app.py` file.
- Adjust the desired level of accuracy and other parameters in the `WaldSPRT` class under the '#settings' comment as needed.
- This work is intended for educational and demonstration purposes.

## References
- # waldSPRT

## Overview
This repository demonstrates the usage of sequential data analysis to predict the binomial population distribution from a small sample. The research combines the techniques of Wald's Sequential Probability Ratio Test (SPRT) and a Binary Search approach to search for the most optimal probability for each sample. The work is inspired by Dr. Manoranjan Sathpathy et al.'s analysis and prediction of Covid-19 cases using the Wald SPRT test in the paper titled "Effect of Migrant Labourer Inflow on the Early Spread of Covid-19 in Odisha: A Case Study".

## Description
The main objective of this project is to estimate the probability of a binomial population distribution based on a small sample using sequential analysis. The Wald SPRT algorithm is implemented to iteratively update the probability estimate until a desired level of accuracy is achieved.

## Features
- Estimation of binomial population distribution probability
- Sequential data analysis using Wald's Sequential Probability Ratio Test
- Binary Search approach for optimal probability search
- Integration with Streamlit for interactive demonstration

## Dependencies
- Python 3.x
- numpy
- matplotlib
- streamlit

## Usage
1. Install the required dependencies using `pip install -r requirements.txt`.
2. Run the Streamlit app by executing `streamlit run test.py`.
3. Access the application through the provided URL in the terminal.

## Additional Notes
- The data used for demonstration can be customized in the `test.py` file.
- Adjust the desired level of accuracy and other parameters in the `WaldSPRT` class as needed.
- This work is intended for educational and demonstration purposes.

## References
- Shreetam Behera, Debi Prosad Dogra, and Manoranjan Satpathy. 2022. Effect of Migrant Labourer Inflow on the Early Spread of Covid-19 in Odisha: A Case Study. ACM Trans. Spatial Algorithms Syst. 8, 4, Article 27 (December 2022), 18 pages. https://doi.org/10.1145/3558778
- A. Wald. "Sequential Tests of Statistical Hypotheses." Ann. Math. Statist. 16 (2) 117 - 186, June, 1945. https://doi.org/10.1214/aoms/1177731118

Please feel free to contribute to this project by opening issues or submitting pull requests.


Please feel free to contribute to this project by opening issues or submitting pull requests.

For any questions or inquiries, please contact [Fabiha Makhdoomi] at [makhdoomifabiha@gmail.com].
