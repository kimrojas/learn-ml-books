# A tour of ML classifiers using Scikit-learn


## Choosing a classification algorithm

- no single classifier works best across all possible scenarios
- Five main steps for training supervised machine learning algorithm
    1. Selecting features and collecting labeled training data
    2. Choosing a performance metric
    3. Choosing  a learning algorithm and training model
    4. Evaluating the performance of the model
    5. Changing the settings of the algorithm and tuning the model

**`train_test_split`**

- **stratify**: returns training and test subsets that have the same proportions of class labels as the input dataset.  

## Modeling class probabilities via logistic regression

### Logistic regression and conditional probabilities

- **odds**: the odds in favor of a particular event $\cfrac{p}{1-p}$, where $p$  is the probability of the positive event or simple $p := p(y=1 | x)$
- **log-odds**: redefining it to a logit function --- $logit(p) = log \cfrac{p}{1-p}$
  - The logit function takes input values in the range 0 to 1 and transforms them into values over the entire real-number range. 
- in the *logistic model*, there is a relationship between weighted inputs and log-odds:

    $logit(p) = w_1x_1 + \cdots + w_mx_m+b = \sum_{i=j} w_jx_j + b = \mathbf{w}^T\mathbf{x} + b$

- inversing the logit function, we have the **sigmoid function**

  $\sigma(z)= \cfrac{1}{1+e^{-z}}$, where z is the net input

### Learning  the model weights via the logistic loss function

The likelihood 

$\mathcal{L} (\mathbf{w}, b | \mathbf{x}) = p(y|\mathbf{x;w},b) $
$\mathcal{L} = \prod_{i=1}^{n} p(y^{(i)}|\mathbf{x^{(i)};w},b)$
$\mathcal{L} = \prod_{i=1}^{n} \left( \sigma(z^{(i)}) \right)^{y^{(i)}} \left( 1- \sigma(z^{(i)}) \right)^{1-y^{(i)}}$ 

The log-likelihood

$\mathcal{l} (\mathbf{w}, b | \mathbf{x}) = \log \mathcal{L} (\mathbf{w}, b | \mathbf{x})$
$\mathcal{l} (\mathbf{w}, b | \mathbf{x}) = \sum_{i=1} \left[y^{(i)} \log (\sigma(z^{(i)})) + (1-y^{(i)}) \log \left(1-\sigma(z^{(i)})\right) \right]$


The log-likelihood as a loss function, $L$

$L(\mathbf{w}, b) = \sum_{i=1}^{n} \left[ -y^{(i)} \log\left(\sigma(z^{(i)})\right) - (1-y^{(i)}) \log\left(1-\sigma(z^{(i)})\right)\right]$

Loss function for one single training example

$L(\sigma(z), y; \mathbf{w}, b) =
\begin{cases}
   -\log (\sigma(z)) &\text{if } y=1 \\
   -\log (1-\sigma(z)) &\text{if } y=0
\end{cases}$

- if prediction is wrong, the loss goes towards infinity. This penalizes wrong predictions with increasingly larger loss.

###   Training a logistic regression model with scikit-learn


- multi-class support is available
- default is 'ovr' or ''multinomial'. Actually these two presents a different result. "multinomial" is the default!
- ovr only works for mutually exclusive targets (meaning it can only belong to one label)
- many other solver exists aside from stochasitc gradient descent (SGD). Most solver will get the convex minimum but some have trouble like the liblinear which cannot be applied to multinomial problems. 


### Tackling overfitting via regularization

- problem: performs well on training data but not on test data (overfitting). in reverse, there is also underfitting where the model is to simple for a complex pattern. 
- To tune the bias-variance (underfitting-overfitting) tradeoff, we can use regularization. 
- Regularization is useful for handling collinearity, filtering noise from data, and prevent overfitting. 
- Concept is to penalize extreme weights.
- Most common is the **L2-regularization** (aka. L2 shrinkage or weight decay) 

$\cfrac{\lambda}{2n} \lVert \mathbf{w} \rVert^2 = \cfrac{\lambda}{2n} \sum_{j=1}^{m} w_j^2$

- here $\lambda$ is the regularization parameter. 
- Sample size $n$ is there to scale the regularization term similar to loss
- regularization requires feature scaling 
- controls how closely we fit the training data while keeping weights small

**loss function with simple regularization term**

$L(\mathbf{w}, b) = \cfrac{1}{n} \sum_{i=1}^{n} \left[ -y^{(i)} \log(\sigma(z^{(i)})) - (1 - y^{(i)}) \log(1 - \sigma(z^{(i)})) \right] + \cfrac{\lambda}{2n} \| \mathbf{w} \|^2
$

**partial derivative with regularizatino term**

$\cfrac{\partial L(\mathbf{w}, b)}{\partial w_j} = \left( \cfrac{1}{n} \sum_{i=1}^{n} \left( \sigma(\mathbf{w}^T \mathbf{x}^{(i)}) - y^{(i)} \right) x_j^{(i)} \right) + \cfrac{\lambda}{n} w_j$

**Parameter C**
- introduced in logistic regression comes from support vector machine conventions
- inversely proportional to regularization parameter $\lambda$
- if regularization strength becomes too high, the weight coeficients aproach zero resulting in underfitting


## Maximum margin classification with support vector machines


- an extension of a perceptron
- in SVM, optimization objective is to maximize the margin
- margin = distance between the separating hyperplane (decision boundary)
- the training examples closest to this hyperplane are called **support vectors**
- intuition is simple but math is advance

### Dealing with a nonlinearly separable case using slack variables

- introduced "slack variables" to allow convergence for nonlinearly separable data in the presence of misclassification
- slack variable introduced parameter "C" in SVM context
- C is the hyperparameter for controlling the penalty for misclassification
- C effectively controls the width of the margin which effectively tunes bias-variance tradeoff

### Alternative implementation in scikit-learn

- Logistics regression use liblinear library
- SVM leverages LIBSVM library, both c++ library for faster computation of multiple classifiers
- sometimes the dataset is too big so an SGD variation of the classifiers are also available (SGDClassifiers). This is based on stochastic gradient descent and also has partial_fit method that we discussed earlier

## Solving nonlinear problems using kervel SVM


