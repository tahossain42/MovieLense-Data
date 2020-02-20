# CSC8101 2019-20 Coursework assignment
This coursework uses the Movielens 20M dataset, specifically the ratings dataset which contains 20,000,263 records.
There are 138,493 unique users, and 26,744 unique movies.

The dataset is available on an Azure Blob store. For convenience, here is Spark code to load the dataset:

import pandas as pd
ratingsURL = 'https://csc8101storageblob.blob.core.windows.net/datablobcsc8101/ratings.csv'
ratings = spark.createDataFrame(pd.read_csv(ratingsURL))
