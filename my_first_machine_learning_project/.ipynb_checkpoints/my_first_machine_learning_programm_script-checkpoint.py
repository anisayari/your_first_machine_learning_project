import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

print('[INFO] Value counts for Survived')
print(train.Survived.value_counts())

#Have the preview of the trianing dataframe. The training dataframe 
#is made for be able to learn from him as a Dev environement.
train.head()