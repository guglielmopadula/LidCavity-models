# Lid Cavity Non intrusive ROM Analysis
In this repo, some common used models are compared on a [Lid](https://github.com/guglielmopadula/LidCavity) [Cavity](https://github.com/guglielmopadula/LidCavity) [Dataset](https://github.com/guglielmopadula/LidCavity).
The models, with their main characteristics and performances, are summed up here.

|   Model     |         Features        |rel u train error|rel u test error|rel v train error|rel v test error|rel p train error|rel p test error| 
|-------------|-------------------------|-----------------|----------------|-----------------|----------------|-----------------|----------------|
|ParMonDMD+RBF|Fixed space and time grid|1.3e-05          |2.9e-05         |1.3e-05          |3.4e-05         |8.0e-06          |6.3e-05         |
|ParParDMD+RBF|Fixed space and time grid|1.1e-02          |1.0e-02         |2.8e-02          |2.6e-02         |1.9e-03          |1.9e-03         |

