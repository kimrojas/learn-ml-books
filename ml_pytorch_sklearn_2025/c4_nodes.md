# Building Good Traning Datasets -- Data Processing

## Dealing with missing data

- we commonly have NaN (not a number) or NULL
- most computational tools cannot handle these so we should do it before further analysis

### Identifying missing values in tabular data

- scikit-learn was primarily designed for use with numpy array 
- nowadays, it more convenient to use pandas for preprocessing so sklearn now also supports this (dataframe object)
- array with the dataframe can always be accessed using the df.values


### Eliminating training examples or features with missing values

- simple to do 
- may remove too much making analysis unreliable. 

### Inputing missing values

- since removing too much results in loss of valuable data, we can use interpolation methods to "fill" in the missing data
- **mean imputation** = replace missing value with the mean value of the entire feature column (can also use with median or most_frequent)


### Understanding the scikit-learn estimator API

- SimpleImputer is part of the **transformer** API in sklearn (used for implementing Python classes related to data transformation [not to be confused with the transformer architecture used in LLM])
- two essential methods of estimators are `fit` and `transform`. 
- `fit` used to learn the parameters of the dataset and `transform` to transform the data based on the parameters. 
- any data array to be transformed needs to have the same number of features as the data array that was used to fit the model
- Classifier in chapter 3 are called **estimators** in sklearn


## Handling categorical data

- need to distinguish between ordinal and nominal features 
- ordinal = categorical features that can be sorted or ordered (t-shirt size)
- nonimal features don't imply order (t-shirt color)
 
### Categorical data encoding with pandas

  
### Mapping ordinal features

- convert categorical strings into integers
- mapping is reversible

### Encoding class labels

- use similar mapping approach
- class labels are not ordinal, meaning it doesnt matter which integer number we assign. 
- Can use sklearn.preprocessing.LabelEncoder

### Performing one-hot encoding on nominal features

- using integer to mapp the class labels leads to a mistake, the AI will now assume that a color is larger than another by comparing the magnitude of the integer mapped to it. 
- **one-hot encoding** - create a new dummy feature for each unique value in the nominal feature column with binary values 
- one-hot encoding introduces colinearity (which can be an issue to some methods, ex, those that require matrix inversion)
- to reduce colinearity, we can remove one feature without loss of data (if the two fetures are not observed, then its the third)
- there are other alternatives to OHE, which are useful for high cardinality (large number of unique labels)
    - Binary encoding (which requires fewer columns)
    - Count / Frequency encoding

**encoding ordinal features**

- if unsure of the numericall difference between categories, we can encode them using a threshold function


## Partitioning a dataset into separate training and test datasets

- comparing predictions to true labels in the test set = unbiased performance evaluation
- after preprocessing, we'll do feature selection to reduce the feature dimensionality
- we'll use the wine dataset (178 examples, 13 features)

- common splits are 60/40, 70/30, 80/20. For very large datasets, 90/10 or 99/1 is enough
- post train/eval, we can retrain using the whole dataset. However, this may result in worse generalization if the test contains outliers or the dataset is small

## Bringing features onto the same scale

- Feature scaling is a crucial step 
- if we have feature1 = (1-10) and feature2 = (100-1000). getting the MSD will be predominantly feature2



