# Lid Cavity Non intrusive ROM Analysis
In this repo, some common used models are compared on a [Lid](https://github.com/guglielmopadula/LidCavity) [Cavity](https://github.com/guglielmopadula/LidCavity) [Dataset](https://github.com/guglielmopadula/LidCavity).

To install the necessary requirements do:

    $ pip install GPy sklearn pydmd scipy torch torchvision torchaudio git+https://github.com/mathLab/EZyRB 


Following Teman (2000) we assume that $u$, $v$ and $p$ belong to the space of functions 
```math
L^{+\infty}(0,10,H^{1}([-0.05,0.05]))=\{u:[0,10]\times [-0.05,0.05]\times [-0.05,0.05]\rightarrow \mathbb{R} \text{ s.t. } \max_[0,10] \int_{[-0.05,0.05]}\sum_{i=1}^{k}\left(|u(t,x,y)|^{2}+|u_{x}(t,x,y)|^{2}+|u_{y}(t,x,y)|^{2}\right)dxdydt<+\infty \},
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
|ParMonDMD+RBF|Fixed space grid and variable time grid with fixed steps|1.1e-05          |3.1e-05         |1.3e-05          |3.6e-05         |2.0e-05          |1.4e-04         |
|ParParDMD+RBF|Fixed space grid and variable time grid with fixed steps|1.4e-02          |1.3e-02         |2.5e-02          |2.3e-02         |8.1e-03          |7.9e-03         |
|ParMonDMD+KNR|Fixed space grid and variable time grid with fixed steps|9.8e-04          |1.5e-03         |1.2e-03          |1.8e-03         |1.8e-03          |2.7e-03         |
|ParParDMD+KNR|Fixed space grid and variable time grid with fixed steps|1.4e-02          |1.3e-02         |2.5e-02          |2.3e-02         |8.9e-03          |8.9e-03         |
|ParMonDMD+GPY|Fixed space grid and variable time grid with fixed steps|1.3e-04          |1.4e-04         |2.3e-04          |2.2e-04         |7.2e-04          |7.1e-04         |
|ParParDMD+GPY|Fixed space grid and variable time grid with fixed steps|1.4e-02          |1.3e-02         |2.5e-02          |2.3e-02         |8.2e-03          |8.0e-03         |
|RNN          |Fixed space grid and variable time grid with fixed steps|8.0e-02          |7.8e-02         |1.6e-02          |1.5e-02         |1.3e-01          |1.2e-01         |
|LSTM         |Fixed space grid and variable time grid with fixed steps|7.0e-02          |6.6e-02         |1.6e-01          |1.5e-01         |1.2e-01          |1.1e-01         |
|GRU          |Fixed space grid and variable time grid with fixed steps|7.3e-02          |6.8e-02         |1.6e-01          |1.5e-01         |9.3e-02          |8.5e-02         |

