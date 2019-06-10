# testing_PINNs
In the context of my Master Thesis I am testing the PINNs algorithm to infer hidden parameters in PDEs using neural networks.

The data is generated with finite differences following https://github.com/barbagroup/CFDPython on
![equation](https://latex.codecogs.com/gif.latex?%5B0%2C2%20%5Cpi%5D%5Ctimes%20%5B0%2C2%20%5Cpi%5D%20%5Ctimes%20%5B0%2CT%5D).
The size of the mesh and T can be inferred from parameters in the options-dictionary ![equation](https://latex.codecogs.com/gif.latex?%28T%20%3D%20%28layers-1%29%5Ccdot%20dt%29).

The amount of points is taken so that we can compare the results with those of https://github.com/Slowpuncher24/mlhiphy_v2 and https://github.com/Slowpuncher24/pde-net-in-tf.

The PDEs are:
* An advection-diffusion equation ![equation](https://latex.codecogs.com/gif.latex?u_t%20%3D%202%28u_x%20&plus;%20u_y%29%20&plus;%200.5%28u_%7Bxx%7D%20&plus;%20u_%7Byy%7D%29)

We provide the PINNs algorithm with the following form of the PDE, in which the coefficients have to be learned:

![equation](https://latex.codecogs.com/gif.latex?u_t%20%3D%20a_u%20%5Ccdot%20u%20&plus;%20a%5Ccdot%20u_x%20&plus;%20b%5Ccdot%20u_y%20&plus;%20c%20%5Ccdot%20u_%7Bxx%7D%20&plus;%20d%20%5Ccdot%20u_%7Byy%7D%20&plus;%20e%5Ccdot%20u_%7Bxy%7D)

* A diffusion equation with nonlinear source ![equation](https://latex.codecogs.com/gif.latex?u_t%20%3D%200.3%20%28u_%7Bxx%7D%20&plus;%20u_%7Byy%7D%29%20&plus;%2015%20%5Ccdot%20%5Csin%28u%29)

Here we provide the PINNs algorithm with the following form of the PDE:

![equation](https://latex.codecogs.com/gif.latex?u_t%20%3D%20c_0%5Ccdot%20%5Csin%28u%29%20&plus;%20c_%7B00%7D%20%5Ccdot%20u%20&plus;%20c_%7B10%7D%20%5Ccdot%20u_x%20&plus;%20c_%7B01%7D%20%5Ccdot%20u_y%20&plus;%20c_%7B20%7D%20%5Ccdot%20u_xx%20&plus;%20c_%7B02%7D%5Ccdot%20u_%7Byy%7D%20&plus;%20c11%20%5Ccdot%20u_%7Bxy%7D%20&plus;%20c30%5Ccdot%20u_%7Bxxx%7D%20&plus;%20c21%5Ccdot%20u_%7Bxxy%7D%20&plus;%20c12%5Ccdot%20u_%7Bxyy%7D%20&plus;%20c03%5Ccdot%20u_%7Byyy%7D)

* The Burgers equation ![equation](https://latex.codecogs.com/gif.latex?u_t%20%3D%20-u%28u_x%20&plus;%20u_y%29%20&plus;%200.3%20%28u_%7Bxx%7D%20&plus;%20u_%7Byy%7D%29)

Here we provide the PINNs algorithm with the following form of the PDE:

![equation](https://latex.codecogs.com/gif.latex?u_t%20%3D%20c_0%5Ccdot%20u%20-%20c_1%5Ccdot%20u_x%5Ccdot%20u%20-%20c_2%5Ccdot%20u_y%5Ccdot%20u%20&plus;%20c_%7B20%7D%5Ccdot%20u_%7Bxx%7D%20&plus;%20c_%7B02%7D%5Ccdot%20u_%7Byy%7D%20&plus;%20c_%7B11%7D%5Ccdot%20u_%7Bxy%7D%20&plus;%20c_%7B30%7D%5Ccdot%20u_%7Bxxx%7D%20&plus;%20c_%7B21%7D%5Ccdot%20u_%7Bxxy%7D%20&plus;%20c_%7B12%7D%5Ccdot%20u_%7Bxyy%7D%20&plus;%20c_%7B03%7D%5Ccdot%20u_%7Byyy%7D)



The PINNs algorithm was developed by Maziar Raissi, Paris Perdikaris, and George Em Karniadakis. It is described in (https://maziarraissi.github.io/PINNs/) and their paper 'Physics Informed Deep Learning (Part II): Data-driven Discovery of Nonlinear Partial Differential Equations' (https://arxiv.org/pdf/1711.10566.pdf).
