# Lid Cavity Non intrusive ROM Analysis
In this repo, some common used models are compared on a [Lid](https://github.com/guglielmopadula/LidCavity) [Cavity](https://github.com/guglielmopadula/LidCavity) [Dataset](https://github.com/guglielmopadula/LidCavity).

To install the necessary requirements do:

    $ pip install GPy sklearn pydmd scipy torch torchvision torchaudio git+https://github.com/mathLab/EZyRB 


We assume that $u$ and $p$ belong to the space of functions 
```math
L^{2}([-0.05,0.05],0,10)=\{u:[0,10]\times [-0.05,0.05]\times [-0.05,0.05]\rightarrow \mathbb{R}^{k} \text{ s.t. } \int_{0}^{10} \int_{[-0.05,0.05]}\sum_{i=1}^{k}\left(|u(t,x,y)\cdot e_{i}|^{2}+|u_{x}(t,x,y)\cdot e_{i}|^{2}+|u_{y}(t,x,y)\cdot e_{i}|^{2}\right)dxdydt<+\infty \},
```
which is a Hilbert space if considering the scalar product
```math
(f,g)=\int_{0}^{10} \int_{[-0.05,0.05]}\sum\limits_{i=1}^{k}\left((f(t,x,y)\cdot e_{i})(g(t,x,y)\cdot e_{i})+(f_{x}(t,x,y)\cdot e_{i})(g_{x}(t,x,y)\cdot e_{i})+(f_{y}(t,x,y)\cdot e_{i})(g_{y}(t,x,y)\cdot e_{i})\right)dxdydt
```



So the relative error that is used for measuring the model performance is:

```math
\left(\frac{\int_{0}^{10} \int_{[-0.05,0.05]}\sum\limits_{i=1}^{k}\left(|(u(t,x,y)-\hat{u}(t,x,y))\cdot e_{i}|^{2}+|(u_{x}(t,x,y)-\hat{u}_{x}(t,x,y))\cdot e_{i}|^{2}+|(u_{y}(t,x,y)-\hat{u}_{y}(t,x,y))\cdot e_{i}|^{2}\right)dxdydt}{ \int_{0}^{10} \int_{[-0.05,0.05]}\sum\limits_{i=1}^{k}\left(|u(t,x,y)\cdot e_{i}|^{2}+|u_{x}(t,x,y)\cdot e_{i}|^{2}+|u_{y}(t,x,y)\cdot e_{i}|^{2}\right)dxdydt}\right)^{\frac{1}{2}}
```


The integrals are calculated with the composite trapezoidal formula.
The derivatives are calculated with first order difference method.

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
|RNN          |Fixed space grid and variable time grid with fixed steps|1.6e-02          |1.4e-02         |2.0e-02          |1.8e-02         |1.6e-02          |1.3e-02         |
|LSTM         |Fixed space grid and variable time grid with fixed steps|1.1e-02          |9.5e-03         |2.6e-02          |2.4e-02         |1.5e-02          |1.2e-02         |
|GRU          |Fixed space grid and variable time grid with fixed steps|1.3e-02          |1.0e-02         |2.2e-02          |2.1e-02         |1.1e-02          |7.9e-03         |

