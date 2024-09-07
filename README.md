# branin-fit-with-mlp

To run, enter `bash run_example.sh` in the terminal. 

A simple example where we fit a rescaled branin function (https://www.sfu.ca/~ssurjano/branin.html) with an MLP consisting of GELU activations. I was curious how well we could fit a smooth function with relatively little data.

![training_features-128_layers-5](https://github.com/user-attachments/assets/9cb9443a-5ec0-4a3b-99e3-09f60d1b0ce6)

Edit: 9/07/2024

Expanded to include a brief comparison of GeLU and ReLU activations. 

RELU:
![example_activation-relu_features-128_layers-5](https://github.com/user-attachments/assets/9d608cd8-5b51-4991-827c-02444edd52ab)

GELU:
![example_activation-gelu_features-128_layers-5](https://github.com/user-attachments/assets/4414c95c-46ed-4c1d-8194-30dc2a86ac66)

Note that if you change the initialization of the ReLU network -- e.g., by initializing the bias in linear layers to 0 -- the quality of fit is similar, but the surface looks considerably different. 

One next step is to examine the impact of stochastic gradient descent on the relu network for this simple example. Naturally, there is a large literature on the "implicit regularization" induced by stochastic/noisy training and a lot of really great work is being done. A few "recent" papers that I have found quite interesting:

Blanc, G., Gupta, N., Valiant, G., & Valiant, P. (2020, July). Implicit regularization for deep neural networks driven by an ornstein-uhlenbeck like process. In Conference on learning theory (pp. 483-513). PMLR.

Andriushchenko, M., Varre, A. V., Pillaud-Vivien, L., & Flammarion, N. (2023, July). Sgd with large step sizes learns sparse features. In International Conference on Machine Learning (pp. 903-925). PMLR.


