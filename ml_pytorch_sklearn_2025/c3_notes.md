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


### kernel method for linearly inseparable data

- kernel methods means to create nonlinear combinations of the original features to project them into a higher-dimensional space via a mapping function $\phi$ where the data becomes linearly separable
- Lets try transforming this using the following transformation:

$\phi(x_1, x_2) = (z_1, z_2, z_3) = (x_1, x_2, x_1^2 + x_2^2)
$

### Using the kernel trick to find separating hyperplanes in high-dimensional space

- problem with mapping to higher dimension is that itis computationally expensive

**kernel function**

$\kappa(\mathbf{x}^{(i)}, \mathbf{x}^{(j)}) = \phi(\mathbf{x}^{(i)})^\top \phi(\mathbf{x}^{(j)})$

**Radial basis function (RBF)** or **Gaussian kernel**

$\kappa(\mathbf{x}^{(i)}, \mathbf{x}^{(j)}) = \exp\left( -\cfrac{\|\mathbf{x}^{(i)} - \mathbf{x}^{(j)}\|^2}{2\sigma^2} \right)
$

$\kappa(\mathbf{x}^{(i)}, \mathbf{x}^{(j)}) = \exp\left( -\gamma \| \mathbf{x}^{(i)} - \mathbf{x}^{(j)} \|^2 \right)
$ where $\gamma = \frac{1}{2\sigma^2}$ the hyperparameter that needs to be optimized

- kernel can be interpreted as **similarity function** between two samples
- 𝛾 parameter also plays an important role in controlling overfitting or variance when the algorithm is too sensitive to fluctuations in the training dataset.

## Decision tree 

- decision algorithm start at tree root
- split data on the feature that results in the largest information gain (IG)
- can repeat splitting procedure at each child node
- this will results in a very pure node (and a very deep tree thats prone to overfitting)
- we can *prune* the tree by setting a limit for maximum depth of the tree

### Maximizing IG - getting the most bang for your buck

split the nodees at the most informative features

objective function to maximize IG

$IG(D_p, f) = I(D_p) - \sum_{j=1}^{m} \cfrac{N_j}{N_p} I(D_j)
$

- $f$ = feature to perform the split
- $D_p$ and $D_j$ are the dataset of the parent and jth child node
- $I$ is our impurity measure
- $N_p$ is the total number of training examples at the parent node
- $IG$ is the difference between the impurity of the parent node and the sum of the child node impurities
- lower the impurities of the child node, the larger the information gain
- For simplicity, libraries like scikit-learn implements binary decision trees, splitting parent node into two child nodes $D_{left}$ and $D_{right}$:

$IG(D_p, f) = I(D_p) - \cfrac{N_{\text{left}}}{N_p} I(D_{\text{left}}) - \cfrac{N_{\text{right}}}{N_p} I(D_{\text{right}})
$

Commonly used impurity measures / splitting criteria 

1. Gini Impurity ($I_G$)
2. entropy ($I_H$)
3. classification error ($I_E$)

**Entropy for all non-empty classes** ($p(i|t) \neq 0$) 

$I_H(t) = -\displaystyle\sum_{i=1}^{c} p(i \mid t) \log_2 p(i \mid t)$

- $p$ is the proportion of the examples that belongs to class $i$ for a particular node $t$
- The entropy is 0 if all examples at a node belongs to the same class
- entropy is maximum if we have a uniform class distribution
- $p(i=1|t) = 1$ or $p(i=0|t) = 0$
- $p(i=1|t) = 0.5$ and $p(i=0|t) = 0.5$ means entropy is max at 1

**Gini impurity**

- Gini is also maximal if the classes are perfectly mixed
- examples, for a binary class (c=1)

$I_G(t) = 1 - \displaystyle\sum_{i=1}^{c} 0.5^2 = 0.5$

- in practice, gini and entropy typically yields similar results
- Its not worth spending time on testing impurity criteria rather than testing different pruning cutoffs

**Classification error**

$I_E(t) = 1 - \max \{ p(i \mid t) \}$

- useful for pruning, not for growing a decision tree


## Building a decision tree

## Combining multiple decision trees via random_forest

- random forest is a ensemble of decision trees
- idea is to average multiple (deep) decision trees that individually suffer  from high variance to build a more robust model with better generalization performance and is less susceptable to overfitting. 
- Algo is
1. draw a random **bootstrap** sample of size $n$ (randomly choose $n$ examples from the training dataset with replacement)
2. Grow a decision tree from the bootstrap sample. At each node:
  a. randomly select $d$ features without replacement
  b. Split the node using the feature that provides the best splits according to the objective function, for instance, maximizing information gain
3. repeat steps 1-2, $k$ times
4. Aggregate the prediction by each tree to assign the class label by **majority vote**. 

> Note: sampling with or without replacement. with replacement, means probability doesn't change (plus the ability to draw the same number more than once). without replacement, probability becomes higher. 


- random forest doesn't have the interpretatibility of decision trees, but is less prone to noise from averaging the predictions of each decision trees
- larger number of trees = better performance of the classifer (at the expense of computational cost)
- hyperparameters are $n$ of the bootstrap sample and the number of features $d$.
- $n$ controls bias-variance. lower $n$ = lower chance that an example is included in the sample. effectively raises randomness and prevents overfitting. too low is also bad cause it lowers performance
- $n$ is usually slightly smaller than the number of training examples
- $d$ is usually smaller than total features. reasonable default is $d = \sqrt{m}$ where m is the number of features of the dataset


## K-nearest neightbhors -- a lazy learning algorithm

- KNN is lazy not because its simple but because it memorizes the training dataset instead

> Parametric vs nonparametric models
> parametric = find a function that can classify new datapoints without the training dataset (SVM, logistic regression, perceptron)
> nonparametric =  instance based learning which memorizes the training dataset. lazy learning = no cost during learning process

**KNN algorithm**

1. choose the number of $k$ and distance metric
2. find the k-nearest neighthor of the data record that we want to satisfy
3. Assign the class label by majority vote

- right choice of k is crucial for bias-variance tradeoff
- KNN is very susceptable to overfitting due to the curse of dimensionality (feature space becomes increasingly sparse for an increasing number of dimensions of a fixed-size training dataset)
- regularization cannot be applied to KNN, so we use feature selection + dimensionality reduction to avoid the curse


