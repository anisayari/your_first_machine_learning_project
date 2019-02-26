# Your First Machine Learning Project
It is a repository to help to you to run your first machine learning algorithm. Build from the famous titanic competitions on Kaggle.

![alt text](https://raw.githubusercontent.com/anisayari/your_first_machine_learning_project/master/images/machine-learning-everywhere.jpg)

https://www.kaggle.com/c/titanic

If you are new in Data Science please do not hesitate to visit this [Medium Article](https://towardsdatascience.com/the-complete-guide-to-start-your-datascience-ai-journey-c3d867215934)

## Description 
The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.

One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.

In this challenge, we ask you to complete the analysis of what sorts of people were likely to survive. In particular, we ask you to apply the tools of machine learning to predict which passengers survived the tragedy.

## Goal of this repo
* Help you to run your first machine learning pipeline in Python.
* Have a really simple explanation to demonstrated the logic used during a Machine Learning project
* Follow the essentials path of any Machine Learning project (import, understanting, visualisation, features engineering, modelisation and prediction)

## What you will learn ?
* import libraries `import`
* read a data file `pandas.read_csv()`
* describe your `pandas.DataFrame()` with `df.info()` and `df.columns_name.value_counts()`
* deal with missing value `df.isna().sum()` and fill a `pandas.Series` with corrected values
* plot figure with `seaborn` using `seaborn.boxplot()` but also display a correlation matrix with `seaborn.heatmap()` and a scatter matrix plot with `seaborn.pairplot`
* create new features and the logic behind this crucial step with a function `def features_engineering():`
* run a model, here we use `xgboost.xgb` and train your model with `xgbosst.xgb.fit()` but also predict value with `xgboost.xgb.predict()`
* Evalue this model with some metric and a cross validation score using `.cross_val_score()`, but also plot features importance with `xgboost.xgb.plot_importances()`
* finally save your prediction to a new file with `pandas.to_csv()

