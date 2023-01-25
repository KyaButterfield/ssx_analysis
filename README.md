# ssx_analysis
Here will be a repository of the code used for SSX data analysis during the 2022 - 2023 academic year. 

Before this iteration of code, SXX data analysis methods were authored by post docs like Tim Gray and Chris Cothran and written in python 2. The code they wrote was rather intrcite and maluable, though without proper documentation (which it certainly lacked) attempts to modernize it in python 3 were futile. 

At the time of writing this readme.md file, the contemporary ssx analysis code is incomplete. We are currently authoring code with the following functionalities in mind:
- Generate code that is well documented (using sphinx or some other platform), user friendly, and readable such that new SSX experimentalists can learn to use it quickly. 
- Generate code that can read in and parse raw data collected using D-TACQ systems (_2022ssxreadin.py to be relabeled: ssxreadin.py)
- Generate attrative line-averaged density vs time plots from Mach Zehnder interferometry data (_2022interferometry.py to be relabeled Mach_Zehnder.py)
- Generate attrative line-averaged density vs time plots from Heterodyne interferometry data (to be created)
- Genetate conpelling and legible 3D magnetic field visualization from magnetic probe array data (_2022magnetics.py to be relabeled: magnetics.py)
- Generate attractve line-averaged temperature vs time plots from ids data (_2022ids.py to be relabeled: ids.py)

