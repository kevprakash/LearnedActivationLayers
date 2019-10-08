# LearnedActivationLayers

Experimental Keras layers for self-learning, self-tuning, and/or approximating activation functions.

Testing is done on several variations of a convolutional model on the CIFAR-10 dataset. Optimizer is RMSProp and loss is sparse categorical cross entropy

Current experimental results:

Tuning Existing Activation Functions with Linear Transformations: A<sub>p<sub>0</sub>, p<sub>1</sub>, p<sub>2</sub>, p<sub>3</sub></sub>(x) = p<sub>0</sub> f(p<sub>1</sub>x + p<sub>2</sub>) + p<sub>3</sub> where f(x) is a standard activation function (ReLu, sigmoid, tanh, etc)
- Always converges when using standard activation functions
- Does not need special learning rate
- Can be learned per layer or can use a single layer multiple times in the network
- Performs better than respective standard activation functions
  - Generally has 5-10 percentage point increase in accuracy for shallow networks
- Can modify range and domain of standard activation functions
- Does not affect learning speed or computation speed significantly

Approximating with Real Fourier Series: A<sub>P, Q</sub>(x) = &Sigma;<sub>n=0</sub> (p<sub>n</sub> cos(x) + q<sub>n</sub> sin(x))
- Frequently does not converge
  - Likely due to periodic nature as well as 0-valued gradients at certain points
- Requires a very low learning rate to consistently converge
- Performs very well **if** it converges
- The more terms, the less likely it is to converge

Approximating with Polynomials: A<sub>P, Q</sub>(x) = &Sigma;<sub>n=0</sub> (p<sub>n</sub>x<sup>n</sup>) / &Sigma;<sub>n=0</sub> (q<sub>n</sub>x<sup>n</sup>)
- Highly unstable
  - Even if it does seem to converge, can suddenly diverge
  - Due to potential 0 value of denominator
- Performs well if using same layer multiple times in network **if** it converges
- Deals with asymptotic behaviour fairly well
- More stable with lower learning rates
