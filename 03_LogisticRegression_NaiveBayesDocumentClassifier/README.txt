Rajesh Sakhamuru 6/30/2020
Assignment 3

My project was developed on Ubuntu 18.04 in Python 3.6.

In order to run the code for this assignment, the following dependencies are necessary:

numpy      1.18.4 :- pip install numpy                   OR  pip3 install numpy
pandas     0.25.3 :- pip install pandas                  OR  pip3 install pandas
matplotlib 2.2.2  :- python -m pip install -U matplotlib OR  python3 -m pip3 install -U matplotlib

In order to run code, please open the terminal and switch directories to the ./src folder.

To run problem 1: Spambase Data:
    - Run the command 'python3 spambaseLogisticRegression.py'
    - Normal Equations vs Gradient Descent can be seen in Fold 1 Output
    - Results will be printed to the console.

To run problem 1: Breast Cancer Data:
    - Run the command 'python3 breastCancerLogisticRegression.py'
    - Normal Equations vs Gradient Descent can be seen in Fold 1 Output
    - Results will be printed to the console.

To run problem 1: Pima Indian Diabetes Data:
    - Run the command 'python3 diabetesLogisticRegression.py'
    - Results will be printed to the console.

To run problem 2: Documents Data:
    - Run the command 'python3 documentsNaiveBayes.py 100'
    - The command line argument '100' can be replaced with '100, 500, 1000, 2500, 5000, 7500 or 10000'
      The datamatrices for the vocab lists of those sizes are saved in the 20NG_data folder so they don't have to be regenerated each run.
      WARNING: Anything other than '100' and '500' take a long time to run, so I advise for testing purposes to just use '100'
               although accuracy for '100' is quite low (it gets much better at around '5000'), it will match my reported results.
    - Results will be printed to the console.

