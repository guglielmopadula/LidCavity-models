# Lid Cavity Non intrusive ROM Analysis
In this repo, some common used models are compared on a [Lid](https://github.com/guglielmopadula/LidCavity) [Cavity](https://github.com/guglielmopadula/LidCavity) [Dataset](https://github.com/guglielmopadula/LidCavity).

To install the necessary requirements do:

    $ pip install GPy sklearn pydmd scipy torch torchvision torchaudio git+https://github.com/mathLab/EZyRB 

The models, with their main characteristics and 
performances, are summed up here.


|   Model     |         Features                                       |rel u train error|rel u test error|rel v train error|rel v test error|rel p train error|rel p test error| 
|-------------|--------------------------------------------------------|-----------------|----------------|-----------------|----------------|-----------------|----------------|
|ParMonDMD+RBF|Fixed space grid and variable time grid with fixed steps|1.6e-10          |7.1-10          |1.8e-10          |6.3e-10         |4.4e-10          |2.3e-08         |
|ParParDMD+RBF|Fixed space grid and variable time grid with fixed steps|6.0e-04          |4.8e-04         |2.6e-03          |2.2e-03         |3.6e-05          |3.4e-05         |
|ParMonDMD+KNR|Fixed space grid and variable time grid with fixed steps|8.2e-06          |7.9e-06         |5.2e-07          |5.9e-06         |1.0e-05          |1.2e-05         |
|ParParDMD+KNR|Fixed space grid and variable time grid with fixed steps|6.0e-04          |4.7e-04         |2.5e-03          |2.1e-03         |4.6e-05          |4.6e-05         |
|ParMonDMD+GPY|Fixed space grid and variable time grid with fixed steps|4.9e-08          |4.0e-08         |4.1e-08          |3.4e-08         |5.1e-07          |4.1e-07         |
|ParParDMD+GPY|Fixed space grid and variable time grid with fixed steps|6.4e-04          |4.8e-04         |2.6e-03          |2.2e-03         |3.6e-05          |3.4e-05         |


