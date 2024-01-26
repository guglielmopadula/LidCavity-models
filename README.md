# Lid Cavity Non intrusive ROM Analysis
In this repo, some common used models are compared on a [Lid](https://github.com/guglielmopadula/LidCavity) [Cavity](https://github.com/guglielmopadula/LidCavity) [Dataset](https://github.com/guglielmopadula/LidCavity).

To install the necessary requirements do:

    $ pip install GPy sklearn pydmd scipy torch torchvision torchaudio git+https://github.com/mathLab/EZyRB 


Following Teman (2000) we assume that $u$, $v$ and $p$ belong to the space of functions 
```math
L^{+\infty}(0,10,H^{1}([-0.05,0.05]))=\{u:[0,10]\times [-0.05,0.05]\times [-0.05,0.05]\rightarrow \mathbb{R} \text{ s.t. } \max_{[0,10]} (\int_{[-0.05,0.05]}\left(|u(t,x,y)|^{2}+|u_{x}(t,x,y)|^{2}+|u_{y}(t,x,y)|^{2}\right)dxdy)^{\frac{1}{2}}<+\infty \},
```



So the relative error that is used for measuring the model performance is:

```math
\left(\frac{\max_{[0,10]} \int_{[-0.05,0.05]}\left(|(u(t,x,y)-\hat{u}(t,x,y))|^{2}+|(u_{x}(t,x,y)-\hat{u}_{x}(t,x,y))|^{2}+|(u_{y}(t,x,y)-\hat{u}_{y}(t,x,y))|^{2}\right)dxdy}{ \max_{[0,10]} \int_{[-0.05,0.05]}\sum\left(|u(t,x,y)|^{2}+|u_{x}(t,x,y)|^{2}+|u_{y}(t,x,y)|^{2}\right)dxdy}\right)^{\frac{1}{2}}
```
We can write it in this way because max and sqrt commute,

The integrals are calculated with the composite trapezoidal formula.
The derivatives are calculated with first order difference method.

The models, with their main characteristics and 
performances, are summed up here.


|   Model     |         Features                                       |rel u train error|rel u test error|rel v train error|rel v test error|rel p train error|rel p test error| 
|-------------|--------------------------------------------------------|-----------------|----------------|-----------------|----------------|-----------------|----------------|
|ParMonDMD+RBF|Fixed space grid and variable time grid with fixed steps|7.2e-06          |7.0e-06         |5.9e-06          |5.6e-06         |2.1e-05          |2.7e-05         |
|ParParDMD+RBF|Fixed space grid and variable time grid with fixed steps|2.7e-03          |2.5e-03         |3.7e-03          |3.4e-03         |3.8e-03          |3.6e-03         |
|ParMonDMD+KNR|Fixed space grid and variable time grid with fixed steps|3.5e-04          |5.5e-04         |5.7e-04          |9.2e-04         |6.8e-05          |9.9e-04         |
|ParParDMD+KNR|Fixed space grid and variable time grid with fixed steps|2.7e-03          |2.6e-03         |3.7e-03          |3.5e-03         |4.0e-03          |3.8e-03         |
|ParMonDMD+GPY|Fixed space grid and variable time grid with fixed steps|6.9e-05          |6.3e-05         |9.9e-05          |8.9e-05         |2.3e-04          |2.3e-04         |
|ParParDMD+GPY|Fixed space grid and variable time grid with fixed steps|2.7e-03          |2.5e-03         |3.7e-03          |3.4e-03         |3.8e-03          |3.6e-03         |
|RNN          |Fixed space grid and variable time grid with fixed steps|1.7e-01          |1.6e-01         |3.0e-01          |2.8e-01         |1.2e-01          |1.1e-01         |
|LSTM         |Fixed space grid and variable time grid with fixed steps|1.4e-01          |1.3e-01         |2.9e-01          |2.7e-01         |1.1e-01          |1.0e-01         |
|GRU          |Fixed space grid and variable time grid with fixed steps|1.4e-01          |1.3e-01         |2.9e-01          |2.7e-01         |6.6e-02          |5.9e-02         |
|NN           |Fixed space and time grid                               |7.0e-02          |6.6e-02         |1.3e-01          |1.2e-01         |1.3e-01          |1.3e-01         |
|TimeNN       |Fixed space and variable time                           |1.2e-01          |1.2e-01         |3.5e-02          |3.5e-02         |5.5e-02          |5.3e-02         | 
|SpaceTimeNN  |Variable space and variable time                        |3.0e-01          |3.0e-01         |4.5e-01          |4.5e-01         |4.8e-01          |4.7e-01         |
|FNO          |Fixed space time grid with superresolution              |2.9e-01          |2.9e-01         |4.9e-01          |4.9e-01         |4.6e-01          |4.5e-01         |