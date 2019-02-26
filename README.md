# New York City Taxi Fare Prediction

[//]: # (Image References)

[image0]: ./pics/regression.png
[image1]: ./pics/MSE.png
[image2]: ./pics/R2.png
[image3]: ./pics/flowchart.png 


## Domain Background

Tourists travel in an unfamiliar city may have a taxi ride, but sometimes they don't know reasonable taxi fare in this city. Few wicked taxi drivers may charge unreasonable fare by sneakily taking long route or adding initial charge.

If tourists have a tool to predict reasonable taxi fare based on some simple features like time, pickup location or dropoff location, they can notice unusual charge, take some actions and prevent from fraud. tourists make their budget on travel expense conveniently. For personal reason, when I have a business trip and have to make a budget in advance, I would use this tool to plan my means of transport. If I have sufficient budget, I can take a taxi for a more comfortable trip. Otherwise, maybe I need to take a train or a bus.

This type of problem is so-called regression problem that demands to predict one continuous target value (e.g. taxi fare) using a set of features. There are many academic research addresses on it : For example, long term travel time is predicted from time, wind speed, temperature ,... etc. features using several state of the art regression methods in [1], Internet slangs for sentiment score is predicted in [2], and sentiment score is predicted using Tweets’ messages in [3].

## Problem Statement
Our target is to predict taxi fare in New York city, and we have several features like pickup GPS location, dropoff GPS location, or number of passengers, etc. to help us build a model to predict. This is a regression problem and we can express it as:

`𝑦 = 𝑓(𝑥0,𝑥1,𝑥2,...)`

where 𝑦 is taxi fare for a ride, 𝑥0, 𝑥1, ... are features like time, GPS location, etc. of this ride, and 𝑓 is a function or model we want to derive.

Given a dataset with many samples having ground truth taxi fare and features, we can apply different machine learning algorithms or even deep neural network to train a model based on them, i.e. finding some set of parameters that can describe the model mathematically. After model is developed, we can predict taxi fare for a given features.

After model is developed, we can evaluate the model performance using certain metric that can describe the value difference between predicted taxi fare `𝑦` and ground truth taxi fare `𝑦_hat`. A simple metric for the problem is mean square error (MSE) that calculates mean square difference of predicted taxi fare and ground truth taxi fare, sum and average them for the number of samples in given dataset. There’re also other metrics, it will be discussed in evaluation metrics section later.

## Datasets and Inputs
In this project, New York City Taxi Fare Prediction [4] dataset provided in Kaggle is used.

- **File description**
	- train.csv - Input features and target fare_amount values for the training set (about 55M rows).
	- test.csv - Input features for the test set (about 10K rows). Our goal is to predict fare_amount for each row.
	- sample_submission.csv - a sample submission file in the correct format (columns key and fare_amount). This file 'predicts' fare_amount to be $11.35 for all rows, which is the mean fare_amount from the training set.

- **Data fields**
	- **ID**
  		- key - Unique string identifying each row in both the training and test sets. 
  	
	- **Input (features)**
		- pickup_datetime - timestamp value indicating when the taxi ride started.
		- pickup_longitude - float for longitude coordinate of where the taxi ride started. 
		- pickup_latitude - float for latitude coordinate of where the taxi ride started.
		- dropoff_longitude - float for longitude coordinate of where the taxi ride ended. 
		- dropoff_latitude - float for latitude coordinate of where the taxi ride ended.
		- passenger_count - integer indicating the number of passengers in the taxi ride.
	- **Output (taxi fare)**
		- fare_amout - float dollar amount of the cost of the taxi ride. This value is only in the
training set


## Solution Statement
Several models will be tried in this project, which are well-known to be good candidates for regression problem [5, 6]. Here are some candidates :

- Polynomial regression model
- SVM regression model
- Random forest regression model
- Multiple layer perceptron (MLP) regression model

Some of them have hyper-parameters to be resolved and grid method will be applied to choose the best one. In MLP regression model, different hidden layers and network structure will be tested. I’ll try some of them to find a best model and for all cases, I use `𝑅^2` as evaluation metric.

## Benchmark Model
A simple benchmark model for this problem is **multiple linear regression model**. Mathematically, it can be expressed as :

![image0]

where 𝜃j is the j-th model parameters to be trained, 𝑥i is the i-th feature value and n is the number of features, 𝜃^T is the transpose of 𝜃 (model’s parameter vector), 𝒙 is the feature vector, and h𝜃 is the hypothesis function using the model parameter 𝜃.
For a simple case, we can use all features (pickup\_datetime, pickup\_datetime, ...) in dataset as
𝑥0, 𝑥2, ... , use fare_amount as 𝑦. By using mean square error function 𝑀𝑆𝐸(𝜃) as cost function
and applying gradient descent method, we can derive a set of model parameter 𝜃 and thus the model.

![image1]

This model will then be applied in the test dataset and 𝑅! will be calculated as the evaluation metric. Because this is a very simple model that generally performs poor in most complex datasets, we can use it as a baseline benchmark. What we designed and trained model should outperform this one in evaluation metric 𝑅^2.


## Evaluation Metrics
The metric used in this project is coefficient of determination `𝑅^2`.
![image2]

where SSE is sum squared error and SST is total sum of squared.

The values for `𝑅^2` range from 0 to 1, which captures the percentage of squared correlation between predicted and actual value of the target variable. Value between 0 and 1 indicates how well the model can explain the data. Higher `𝑅^2` is, better the model is. Note that a model can be given `𝑅^2` < 0, which means it’s arbitrarily worse than always predicts the mean of the target variable.

## Project Design
Here’s the flow chart of this project :

![image3]

Step 1 : Explore data and find features’ relationship

Step 2 : Preprocess data such as : impute missing field, transform data, split data into train/valid/test data with data shuffling, feature selection/extraction, feature scaling, outlier removal, ...

Step 3 : Select and implement a model such as SVM regression model, Random Forest regression model, Multiple Layer Perceptron regression model ... to train using train data and valid data

Step 4 : Evaluate model using test data

## Reference
- [1] https://www.researchgate.net/publication/230819938\_Comparing_state-of-the-art\_regression\_meth ods\_for\_long\_term\_travel\_time\_prediction/download
- [2] https://www.researchgate.net/publication/283318703\_Detection\_and\_Scoring\_of\_Internet\_Slangs\_for\_Sentiment\_Analysis\_Using\_SentiWordNet
- [3] https://www.researchgate.net/deref/http%3A%2F%2Faclweb.org%2Fanthology%2F%2FS%2FS13 %2FS13-2053.pdf
- [4] https://www.kaggle.com/c/new-york-city-taxi-fare-prediction#description
- [5] Python Machine Learning 2nd edition, by Sebastian Raschka and Vahid Mirjalili
- [6] Hands-On Machine Learning with Scikit-Learn & TensorFlow, by Aurelien Geron
- [7] https://beamandrew.github.io/deeplearning/2017/02/23/deep\_learning\_101\_part2.html


## File Hierarchy
		New York City Taxi Fare Prediction.ipynb
		New York City Taxi Fare Prediction.pdf
		README.md
		proposal.pdf
		report.pdf
		new-york-city-taxi-fare-prediction
		|- test.csv
		|- train.csv
		|- sample_submission.csv 

## Dataset

https://www.kaggle.com/c/new-york-city-taxi-fare-prediction/data

Download data from this [link](https://www.kaggle.com/c/10170/download-all), extract it and place the extracted folder `new-york-city-taxi-fare-prediction` as file hierarchy shown above.

## Library
- numpy
- pandas
- sklearn
- seaborn
- matplotlib
- jupyter notebook
- scipy
