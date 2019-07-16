# Hyperparameters tuning

> Don’t spend too much time tuning hyperparameters， Only if you don’t have any more ideas or you have spare computational resources

## General pipeline

1. Select the most influential parameters
- There are tons of parameters and we can’t tune all of them
2. Understand, how exactly they influence the training
- A parameter in red
    – Increase it to underfitting
    – Increasing it impedes fitting
    – Decrease to allow model fit easier
- A parameter in green
    - Increasing it leads to a better fit(overfit) on trainset
    - Increase it,if model underfits
    - Decreaseifoverfits
3. Tune them!
- Manually (change and examine)
- Automatically(hyperopt,etc.)
    – Hyperopt
    – Scikit-optimize 
    – Spearmint
    – GPyOpt
    – RoBO
    – SMAC3

## Tree based model

### GBDT

|XGBoost|LightGBM|
|:--|:--|
| <font color="blue">max_depth</font> |  <font color="blue">max_depth/num_leaves</font>  |
| <font color="blue">subsample</font>  |  <font color="blue">bagging_fraction</font>  |
| <font color="blue">colsample_bytree<font>/<font color="blue">colsample_bylevel</font>  | <font color="blue">feature_fraction</font>   |
| <font color="red">min_child_weight </font> |  <font color="red">min_data_in_leaf</font>  |
| <font color="red">lambda, alpha</font>  |  <font color="red">lambda, alpha</font>  |
| <font color="blue"> eta </font> |  <font color="red">learning_rate</font>  |
| <font color="blue">num_round</font>  | <font color="blue">num_iterations</font>   |
| seed  | seed  |


### RandomForest/ExtraTrees
- <font color="yellow">N_estimators (the higher the better)</font>
- <font color="blue">cmax_depth</font>
- <font color="blue">cmax_features</font>
- <font color="red">cmin_samples_leaf</font>
- criterion ('gini' is better in most of time)
- random_state
- n_jobs


### Neural Nets
- <font color="blue">Number of neurons per layer</font>
- <font color="blue">Number of layers</font>
- Optimizers
    – <font color="red">SGD + momentum</font>
    – <font color="blue">Adam/Adadelta/Adagrad/...</font>
        - In practice lead to more overfitting
- <font color="blue">Batch size</font>
- Learning rate (not too high or not too low, depend on other parameters)
- Regularization
    – <font color="blue">L2/L1 for weights</font>
    – <font color="blue">Dropout/Dropconnect</font>
    – <font color="blue">Static dropconnect</font>

### Linear Model
<font color="red">Regularization parameter (C, alpha, lambda, ...)</font>
– Start with very small value and increase it.
– SVC starts to work slower as C increases

Regularization type
– L1/L2/L1+L2 -- try each
– L1 can be used for feature selection


# Practical Guide for competition

## Define your goals. 
What you can get out of your participation?
1. To learn more about an interesting problem
2. To get acquainted with new software tools
3. To hunt for a medal

## Working with ideas
1. Organize ideas in some structure
2. Select the most important and promising ideas
3. Try to understand the reasons why something does/doesn’t work

### Initial pipeline
1. Get familiar with problem domain
2. Start with simple (or even primitive) solution
3. Debug full pipeline
− From reading data to writing submission file
4. “From simple to complex”
− I prefer to start with Random Forest rather than Gradient Boosted Decision Trees

### data loading
1. Do basic preprocessing and convert csv/txt files into hdf5/npy for much faster loading
2. Do not forget that by default data is stored in 64-bit arrays, most of the times you can safely downcast it to 32-bits
3. Large datasets can be processed in chunks

### Performance evaluation
1. Extensive validation is not always needed
2. Start with fastest models - such as LightGBM

## Everything is a hyperparameter
Sort all parameters by these principles: 
1. Importance
2. Feasibility
3. Understanding

Note: changing one parameter can affect the whole pipeline

![](resources/1.png)


## tricks
1. Fast and dirty always better
- Don’t pay too much attention to code quality
- Keep things simple: save only important things
- If you feel uncomfortable with given computational resources - rent a larger server
2. Use good variable names
- If your code is hard to read — you definitely will have
problems soon or later
3. Keep your research reproducible
- Fix random seed
− Write down exactly how any features were generated
− Use Version Control Systems (VCS, for example, git)
4. Reuse code
− Especially important to use same code for train and test stages
5. Read papers
- For example, how to optimize AUC
6. Read forums and examine kernels first
7. Code organization
- keeping it clean
- macros
- test/val

## Pipeline detail

|Procedure| days|
|:--|:--|
|Understand the problem| 1 ~ 2 days|
|Exploratory data analysis | 1 ~ 2 days|
|Define cv strategy| 1 day|
|Feature Engineering| until last 3 ~ 4 days|
|Modeling| Until last 3 ~ 4 days|
|Ensembling| last 3 ~ 4 days|


### Understand broadly the problem
1. type of problem
2. How big is the dataset
3. What is the metric
4. Previous code revelant
6. Hardware needed (cpu, gpu ....)
7. Software needed (TF, Sklearn, xgboost, lightgBM)


### EDA
see the blog [Exploratory data analysis](https://zhangruochi.com/Exploratory-data-analysis/2019/07/13/)

### Define cv strategy
1. This setp is critical
2. Is time is important? **Time-based validation**
3. Different entities than the train. **StratifiedKFold Validation**
4. Is it completely random? Random validation
5. Combination of all the above
6. Use the leader board to test

### Feature Engineering
![](resources/2.png)

### Modeling 
![](resources/3.png)


### Ensembling
![](resources/4.png)



# Advanced Features

## Statistics and distance based features

### Groupby feature

```python
gb = df.groupby(['user_id','page_id'], as_index = False).agg({'ad_price': {'max_price': np.max, 'min_price': np.min}})
gb.columns = ['user_id','page_id','min_price','max_price']
```
![](resources/5.png)

- How many pages user visited
- Standard deviation of prices
- Most visited page
- Many, many more

### Neighbors
- Explicit group is not needed
- More flexible
- Much harder to implement

such as:

- Number of houses in 500m, 1000m,..
- Average price per square meter in 500m, 1000m,..
- Number of schools/supermarkets/parking lots in 500m, 1000m,..
- Distance to closest subway station

KNN features as example:
- Mean encode all the variables
- For every point, find 2000 nearest neighbors using Bray-Curtis metric
$$\frac{\sum|\mu_i - \upsilon_i|}{|\mu_i + \upsilon_i|}$$
- Calculate various features from those 2000 neighbors
    - Mean target of nearest 5,10,15,500, 2000 neighbors
    - Mean distance to 10 closest neighbors
    - Mean distance to 10 closest neighbors with target 1
    - Mean distance to 10 closest neighbors with target 0

## Matrix Factorizations for Feature Extraction

- Matrix Factorization is a very general approach for dimensionality reduction and feature extraction
- It can be applied for transforming categorical features into real-valued
- Many of tricks trick suitable for linear models can be useful for MF

![](resources/6.png)
![](resources/7.png)

- Can be apply only for some columns
- Can provide additional diversity
    − Good for ensembles
- It is a lossy transformation. Its’ efficiency depends on:
    − Particulartask
    − Numberoflatentfactors(Usually 5-100)
- Several MF methods you can find in sklearn
- SVD and PCA
    − Standart tools for Matrix Factorization
- TruncatedSVD
    − Works with sparse matrices
- Non-negative Matrix Factorization (NMF)
    − Ensures that all latent factors are non-negative 
    − Good for counts-like data

## Feature interactions
![](resources/8.png)
![](resources/9.png)
![](resources/10.png)

- We have a lot of possible interactions − N\*N for N features
    - Even more if use several types in interactions
- Need to reduce its’ number
    - Dimensionality reduction
    - Feature selection
- Interactions' order
    - We looked at 2nd and higher order interactions.
    - It is hard to do generation and selection automatically.
    - Manual building of high-order interactions is some kind of art.


### Frequent operations for feature interaction
- Multiplication
- Sum
- Diff
- Division

### Example of interaction generation pipeline
![](resources/11.png)


### Extract features from DT
![](resources/12.png)
get the index of the leaf that each sample is predicted as. it a method to get the high order features

```python
tree.apply()
```

## tSNE

- Result heavily depends on hyperparameters (perplexity)
- Good practice is to use several projections with different perplexities (5-100)
- Due to stochastic nature, tSNE provides different projections even for the same data\hyperparams
    − Train and test should be projected together
- tSNE runs for a long time with a big number of features
    − it is common to do dimensionality reduction before projection.


# Ensemble methods

> Ensemble methods means combining different machine learning models to get a better prediction

## Averageing (or blending)

$$(model_1 + model_2)/2$$

## Weighted averaging

$$model_1 * 0.3 + model_2 * 0.7$$

## Conditional averaging

$$model_1 \, if \, x < 50 \, else \, model_2 $$

## Bagging

### What is bagging
Bagging means **averaging** slightly different versions of the **same model** to improve accuracy



## Boosting

### What is Boosting
A form of weighted averaging of models where each model is built sequentially via taking into account the past model performance

### Main boosting types
1. Weight based
2. Residual based

#### Weighted based
![](13.png)

#### Weighted based boosting parameters
- Learning rate
- Number of estimators
- Input model - can be anything that accepts weights
- Sub boosting type:
    - AdaBoost
    - LogitBoost

#### Residual based boosting
![](14.png)
![](15.png)

we use the error to get the **direction**, and update our prediction through that direction

### Residual based boosting parameters
- Learning rate
- Number of estimators
- Row sampling
- Column (sub) sampling
- Input model - better be trees
- Sub boosting type:
    - Fully gradient based
    - Dart
- Implementation 
    - XGBoost
    - LightGBM
    - H2O's GBM
    - Catboost
    - Sklearn's GBM
    
## Stacking
## StackNet

