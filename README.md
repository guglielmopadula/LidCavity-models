# Lid Cavity Non intrusive ROM Analysis
In this repo, some common used models are compared on a [Lid](https://github.com/guglielmopadula/LidCavity) [Cavity](https://github.com/guglielmopadula/LidCavity) [Dataset](https://github.com/guglielmopadula/LidCavity).

To install the necessary requirements do:

    $ pip install GPy sklearn pydmd scipy git+https://github.com/mathLab/EZyRB 

The models, with their main characteristics and 
performances, are summed up here.


|   Model     |         Features                                       |rel u train error|rel u test error|rel v train error|rel v test error|rel p train error|rel p test error| 
|-------------|--------------------------------------------------------|-----------------|----------------|-----------------|----------------|-----------------|----------------|
|ParMonDMD+RBF|Fixed space grid and variable time grid with fixed steps|1.3e-05          |2.9e-05         |1.3e-05          |3.4e-05         |8.0e-06          |6.3e-05         |
|ParParDMD+RBF|Fixed space grid and variable time grid with fixed steps|1.1e-02          |1.0e-02         |2.8e-02          |2.6e-02         |1.9e-03          |1.9e-03         |
|ParMonDMD+KNR|Fixed space grid and variable time grid with fixed steps|2.4e-03          |2.3e-03         |2.4e-03          |2.5e-03         |3.1e-03          |2.6e-03         |
|ParParDMD+KNR|Fixed space grid and variable time grid with fixed steps|1.2e-02          |1.1e-02         |2.8e-02          |2.6e-02         |3.6e-03          |3.2e-03         |
|ParMonDMD+GPY|Fixed space grid and variable time grid with fixed steps|2.4e-04          |2.2e-04         |2.5e-04          |2.3e-04         |1.1e-03          |9.5e-04         |
|ParParDMD+GPY|Fixed space grid and variable time grid with fixed steps|1.1e-02          |1.0e-02         |2.8e-02          |2.7e-02         |1.9e-03          |1.9e-03         |


