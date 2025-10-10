# Chapter 2 Notes

**Formal definition of artificial neuron**

For a simple binary classification, we can define a decision function $\sigma(z)$, that takes linear combination of certain input values, $x$, and a corresponding weight vector $w$. 

- $z =  w_1x_1 + w_2x_2 + \cdots w_mx_m$
- $\mathbf{w} = \begin{bmatrix} w_1 \\\ \vdots \\\ w_2  \end{bmatrix}$
- $\mathbf{x} = \begin{bmatrix} x_1 \\\ \vdots \\\ x_2  \end{bmatrix}$

if the net input of a particular example $\mathbf{x}^{(i)}$ is greater than a defined threshold, we predict class 1, otherwise class 2. 

In perceptron algorithm, the decision function, $\sigma$ is a variant of a **step function**

$\sigma{(z)} = \begin{cases}
   1 &\text{if } z \ge \theta \\
   0 &\text{otherwise} 
\end{cases}$

## **The perception learning rule**

1. initialize weights and bias unit to 0 or small random numbers
2. For each training example $\mathbf{x}^{(i)}$:
   1. Compute output value $\hat{y}^{(i)}$
   2. Update the weights and bias unit


         $w_j := w_j + \Delta w_j$
         $b := b + \Delta b$

$\Delta w_j = \eta (y^{(i)} - \hat{y}^{(i)}){x_j}^{(i)}$ and $\Delta b = \eta (y^{(i)} - \hat{y}^{(i)})$


Note: 
- $\eta$ is the learning rate typically ranging from 0 to 1
- $y^{(i)}$ is the **true class label** of the $i$th example. 
- $\hat{y}^{(i)}$ is the **predicted class label**
- **Updates happens simultaneously**: we don't recompute $\hat{y}^{(i)}$ before bias unit and weights have been updated. 
- Convergence of perceptron is only guaranteed if the two classes are linearly separable. This means we need to set a maximum number of passes (**epoch**) and/or a tolerance threshold, otherwise it won't stop calculating.   

## Adaptive linear neurons and the convergence of learning (Adaline)

- Key difference is that the weights and biases are updated based on linear activation, instead of a step function. 
- The threshold (step) function is still there to make the final prediction

**Minimizing loss with gradient descent**

- One of the key ingredients of ML is the **Objective function** (loss/cost function)
- In Adalene, we use the **mean squared error (MSE)** between calculated outcome and the true class label:
   $L (\mathbf{w},b) = \frac{1}{2n} \sum_{i=1}^{n}(y^{(i)} - \sigma({z^{(i)}}))^2$
- This is convex shaped hence gradient descent can be used.
- Update to parameters are now possible by taking a step in the opposite step of the direction of the gradient, $\nabla{L}(\mathbf{w}, b)$, of our loss function

   $\mathbf{w} := \mathbf{w} + \Delta\mathbf{w}$
   $b := b + \Delta b$

- The parameter definition are now different. $\Delta\mathbf{w}$ and $\Delta b$ are now the negative gradient multiplied by the learning rate

   $\Delta\mathbf{w} = -\eta \nabla_w{L}(\mathbf{w},b)$
   $\Delta b = -\eta \nabla_b{L}(\mathbf{w},b)$

- The gradient is computed by getting the partial derivatives with respect to $w_j$

   $\frac{\partial{L}}{\partial{w_j}} = -\frac{2}{n}\sum_i{(y^{(i)} - \sigma({z^{(i)}}))x_j^{(i)}}$

- partial for bias is 

   $\frac{\partial{L}}{\partial{b}} = -\frac{2}{n}\sum_i{(y^{(i)} - \sigma({z^{(i)}}))}$

- we can then write the updates as

   $\Delta w_j = -\eta \frac{\partial{L}}{\partial{w_j}}$
   $\Delta b = -\eta \frac{\partial{L}}{\partial{b}}$

- Though similar, Adaline now has a float net input z, instead of the original integer
- Weight updates is based upon all training examples and reffered to as **batch gradient descent** or **full batch gradient descent**

**Improving gradient descent though feature scaling**

- **standardization**: a type of feature scaling that helps gradient descent learning converge faster; however does not make the original dataset normally distributed. 
   - shifts the mean so it is centered at zero
   - each feature has a standard deviation of 1 (unit variance)
   - Example, for $j$th feature, with sample mean $\mu_j$ and standard deviation, $\sigma_j$:
   $x'_j = \frac{x_j- \mu_j}{\sigma_j}$
   - Here $x_j$ is a vector consisting of the $j$th feature value for all training examples, $n$
   - Reason this helps is because the learning rate might be okay with one feature but may also be too big/small for another. 


**Large-scale machine learning and stochastic gradient descent**

- *full batch gradient descent* will be computationally costly to use in datasets containing millions of data. 
- An alternative is the **stochastic gradient descent (SGD)** which is also termed *iterative* or *online* gradient descent
- Instead of updating the weights based on the sum of accumulated errors over all training examples, **we update the parameters incrementally for each training example**

   $\Delta w_j = \eta \left( y^{(i)} - \sigma{(z^{(i)})}\right) x_j^{(i)}$
   $\Delta b = \eta \left( y^{(i)} - \sigma{(z^{(i)})}\right) 
   $
- Typically converges faster due to more frequent weight updates
- gradient is noisier due to being computed from a single training example.
- It is important to **Shuffle** the training data in random order for every epoch to prevent *cycles*
- **Online learning** means we can train on the fly as new data arrives. Old training examples can also be removed and replaced by the new training examples while retaining the pre-trained model (with its parameters)


> **Note**: SGD implementations often uses an adaptive learning rate which decreases based on the number of iterations/epochs. 
> $\frac{c1}{[number of iterations]+c2}$
> - Using an adaptive learning rate allows for further annealing near minima
>

> **Mini-batch gradient descent**

