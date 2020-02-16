# Databricks notebook source
# MAGIC %md TASK 1

# COMMAND ----------

import pandas as pd
ratingsURL = 'https://csc8101storageblob.blob.core.windows.net/datablobcsc8101/ratings.csv'
ratings = spark.createDataFrame(pd.read_csv(ratingsURL))

# COMMAND ----------

ratings.count()

# COMMAND ----------

ratings.show(5)

# COMMAND ----------

# To calculate the average number of ratings per user, the ratings dataframe is grouped by the userId and the entries per user counted. 
# In the next command the created dataframe with number of ratings per user is then used to calculate the average. 
from pyspark.sql.functions import avg
ratingsPerUser = ratings.select("userId", "rating").groupBy("userId").count()
ratingsPerUser.show(5)

# COMMAND ----------

ratingsPerUser.agg({"count": 'avg'}).show()

# COMMAND ----------

# Calculating the average number of ratings per movie is similarly done to the average number of ratings per user, only that the ratings dataframe is this time grouped by the movieId. 
ratingsPerMovie = ratings.select("movieId", "rating").groupBy("movieId").count()
ratingsPerMovie.show(5)

# COMMAND ----------

ratingsPerMovie.agg({"count": 'avg'}).show()

# COMMAND ----------

# Creating the two dataframes 'ratingsPerUser' and 'ratingsPerMovie' simplify the creation of a histogram showing the distribution of the ratings. 
display(ratingsPerUser)

# COMMAND ----------

display(ratingsPerMovie)

# COMMAND ----------

# MAGIC %md TASK 2

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator

# COMMAND ----------

ratings = ratings.drop('timestamp')

# COMMAND ----------

(train, test) = ratings.randomSplit([0.50, 0.50], seed = 1234)

# COMMAND ----------

als = ALS(userCol="userId", itemCol="movieId", ratingCol="rating",
          nonnegative = True, implicitPrefs = False,
          coldStartStrategy = 'drop')

evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")

# COMMAND ----------

paramGrid = ParamGridBuilder() \
           .addGrid(als.rank, [10, 5, 1]) \
           .addGrid(als.regParam, [0.001,0.01,0.1])\
           .build()

# COMMAND ----------

cv = CrossValidator(
  estimator= als, estimatorParamMaps=paramGrid,
  evaluator=evaluator, numFolds=3)

# COMMAND ----------

cvmodel = cv.fit(train)
bestModel = cvmodel.bestModel

# COMMAND ----------

bestModel.rank

# COMMAND ----------

predictions = bestModel.transform(test)
evaluation = evaluator.evaluate(predictions)
print(evaluation)

# COMMAND ----------

display(predictions)

# COMMAND ----------

# MAGIC %md TASK 3

# COMMAND ----------

import pandas as pd
import pyspark.sql.functions as f
from pyspark.sql.types import *
from pyspark.sql import Row
from operator import add
from collections import Counter

# COMMAND ----------

sampleUsers = ratings.select("userId").distinct().sample(fraction=0.1)
sampleUsers.count()

# COMMAND ----------

from pyspark.sql.functions import when
ratings_small = ratings[ratings["userId"].isin(sampleUsers.toPandas()["userId"].tolist())]
ratings_small.count()

# COMMAND ----------

ratings_small_parquet = "/FileStore/tables/190520304/ratings-small.parquet"

# COMMAND ----------

# MAGIC %md TASK 4

# COMMAND ----------

# IN File locations
ratings_small_parquet = "/FileStore/tables/190520304/ratings-small.parquet"

# OUT file locations
edges_small_parquet = "/FileStore/tables/190520304/edges-small.parquet"

# COMMAND ----------

ratings_small = ratings_small.drop('timestamp')
ratings_small = ratings_small.drop('rating')

# COMMAND ----------

ratings_small = spark.read.parquet("rating-small.parquet")
df1 = ratings_small.select("movieId", "userId")
df2 = ratings_small.select("movieId", "userId").withColumnRenamed("movieId", "movieId2").withColumnRenamed("userId", "userId2")
UserMap = df1.join(df2, df1.movieId == df2.movieId2)

# COMMAND ----------

UserMap = UserMap.select("userId", "userId2")
UserMap.show(5)

# COMMAND ----------

UserMap.count()

# COMMAND ----------

from pyspark.sql.functions import col, sum
edgesWithDuplicates = UserMap.groupBy(UserMap.columns).count().filter(UserMap.userId != UserMap.userId2)
edgesWithDuplicates.orderBy("count", ascending=False).show(5)

# COMMAND ----------

edgesWithDuplicates.count()

# COMMAND ----------

# Set User 1 to the smaller id and user2 to the larger id.... then remove duplicates
edgesRDD = edgesWithDuplicates.select("userId", "userId2", "count").rdd.map(lambda x: (min(x["userId"], x["userId2"]), max(x["userId"], x["userId2"]), x["count"]))

# COMMAND ----------

edges1 = spark.createDataFrame(edgesRDD, ["src", "dst", "count"])
edges = edges1.dropDuplicates()
edges.count()

# COMMAND ----------

edges.agg({"count": 'avg'}).show()

# COMMAND ----------

edgesFiltered = edges.filter("count>15")
edgesFiltered.count()

# COMMAND ----------

verticesList = edgesFiltered.select("src").distinct().toPandas().values.tolist() + edgesFiltered.select("dst").distinct().toPandas().values.tolist()
verticesFiltered = spark.createDataFrame(verticesList, ["id"]).distinct()
verticesFiltered.show(5)

# COMMAND ----------

verticesFiltered.count()

# COMMAND ----------

import os,sys
import pyspark.sql.functions as f

# COMMAND ----------

from functools import reduce
from pyspark.sql.functions import col, lit, when
from graphframes import*

# COMMAND ----------

graph = GraphFrame(verticesFiltered, edgesFiltered)
display(graph.edges)

# COMMAND ----------

# MAGIC %md TASK 5

# COMMAND ----------

sc.setCheckpointDir("dbfs:/tmp/group1/checkpoint")
connectedComp = graph.connectedComponents()
display(connectedComp.orderBy("component", ascending = False))

# COMMAND ----------

#maxComp = connectedComp.filter(connectedComp.component == connectedComp.agg({"component": 'max'}).collect()[0]["max(component)"])
#maxComp.show()

# COMMAND ----------

g2 = (connectedComp.describe("component").filter("summary = 'max'").select("component").collect()[0].asDict()['component'])

# COMMAND ----------

g2 = graph.connectedComponents()


# COMMAND ----------

display(connectedComp.orderBy("component", ascending = False))

# COMMAND ----------

# MAGIC %md TASK 6

# COMMAND ----------

import networkx as nx
import numpy as np
import pandas as pd

# COMMAND ----------

edgesFiltered.show()

# COMMAND ----------

verticesFiltered.show()

# COMMAND ----------

src = edgesFiltered.toPandas()["src"].tolist()

# COMMAND ----------

dst = edgesFiltered.toPandas()["dst"].tolist()

# COMMAND ----------

edges_new = pd.DataFrame()
edges_new['src'] = src
edges_new['dst'] = dst
edges_new[['src','dst']]
G=nx.from_pandas_edgelist(edges_new,'src','dst')

# COMMAND ----------

G.number_of_nodes()
list(G.nodes)