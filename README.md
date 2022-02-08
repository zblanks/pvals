# Overview and Testing Details
These repository contains an implementation for a simple ALS algorithm,
and then tests it on separating two mixed speeches.
To use and test this implementation on your own machine,
take the following steps:

1. `git clone https://github.com/zblanks/pvals.git` to your desired location
2. Ensure you have Python and Poetry installed on your machine
    * See https://python-poetry.org/docs/ for instructions on how to get Poetry installed
3. Navigate to the directory containing the PVALS repository
4. `poetry install`
    * This will build the dependencies 
    and install the pvals package in the virtual environment
5. `poetry shell`
    * This spawns a .venv shell allowing you to run this experiment
6. To run the signal separation experiment
type: `python experiment.py`
    * This will use the default noise value of 1,
    a standard Gaussian initialization for all matrices,
    and produce a loss plot
    * You can see more details on command line arguments by typing `python experiment.py -h`

And that's it! I summarized my experimental details in the submitted homework,
but feel free to play around with this implementation on your own.
