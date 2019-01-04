          *** A BRIEF DESCRIPTION OF THE CONTENTS OF THIS FOLDER ***

**************************** Functions.py **************************************
This file contains all the functions that we have implemented. It is divided in 
the following 4 parts:

 - POINT A: functions to implement the basic MC algorithm
 - POINT B: functions to implement the adaptive time step MC
 - POINT C: functions to achieve a variance reduction (antithetic variables
            and MLMC)
 - POINT D: splitting method in order to estimate rare events probabilities


**************************** Parameters.py *************************************
A set of parameters that defines our setting.


************************** Notebooks (folder) **********************************
In this folder you can find 3 Jupyter notebooks.

 - PDE_solver.ipynb: shows the procedure to solve the PDE system used in the 
        project. At the end of this notebooks the probability at some points is
        also computed.

 - MLMC.ipynb and Splitting_method.ipynb: these 2 notebooks contain examples of
        how to use functions that are coded in the sections POINT C and POINT D
        of 'functions.py'. 
        We did not prepared notebooks to also show the usage of the functions 
        implemented in the first 2 sections, since the utilization of those 
        functions is quite straightforward.