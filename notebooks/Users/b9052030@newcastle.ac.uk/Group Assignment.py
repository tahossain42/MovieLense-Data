# Databricks notebook source
# MAGIC %md #TASK 1

# COMMAND ----------

#importing the dataset
import pandas as pd
ratingsURL = 'https://csc8101storageblob.blob.core.windows.net/datablobcsc8101/ratings.csv'
ratings = spark.createDataFrame(pd.read_csv(ratingsURL))

# COMMAND ----------

# MAGIC %md ## Data Exploration

# COMMAND ----------

# total number of data
ratings.count()

# COMMAND ----------

# MAGIC %md ###Graphical and Numerical Summaries

# COMMAND ----------

# MAGIC %md #### Ratings Histogram

# COMMAND ----------

display(ratings)

# COMMAND ----------

# MAGIC %md #### Average number of ratings per users

# COMMAND ----------

# To calculate the average number of ratings per user, the ratings dataframe is grouped by the userId and the entries per user counted. 
# In the next command the created dataframe with number of ratings per user is then used to calculate the average. 
from pyspark.sql.functions import avg
ratingsPerUser = ratings.select("userId", "rating").groupBy("userId").count()
ratingsPerUser.show(5)

# COMMAND ----------

ratingsPerUser.agg({"count": 'avg'}).show()

# COMMAND ----------

# MAGIC %md #### Average number of ratings per movie

# COMMAND ----------

# Calculating the average number of ratings per movie is similarly done to the average number of ratings per user, only that the ratings dataframe is this time grouped by the movieId. 
ratingsPerMovie = ratings.select("movieId", "rating").groupBy("movieId").count()

# COMMAND ----------

ratingsPerMovie.agg({"count": 'avg'}).show()

# COMMAND ----------

# MAGIC %md #### Histogram of the distribution of movie ratings per user

# COMMAND ----------

# Creating the two dataframes 'ratingsPerUser' and 'ratingsPerMovie' simplify the creation of a histogram showing the distribution of the ratings. 
display(ratingsPerUser)

# COMMAND ----------

# MAGIC %md #### Histogram showing the distribution of movie ratings per user

# COMMAND ----------

display(ratingsPerMovie)

# COMMAND ----------

# MAGIC %md # TASK 2

# COMMAND ----------

#loading necessary library
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator

# COMMAND ----------

# MAGIC %md #### Data Cleaning

# COMMAND ----------

# timestamp column is dropped as it will not be useful for the rest of analysis.
ratings = ratings.drop('timestamp')

# COMMAND ----------

# MAGIC %md #### Test Train split

# COMMAND ----------

# The data will be split into a trainning for the fit and a testing set for the evaulation later. A 50% 50% train and test split is used here as with the rest of the models used for this task.
(train, test) = ratings.randomSplit([0.50, 0.50], seed = 1234)

# COMMAND ----------

# MAGIC %md #### ALS Construction

# COMMAND ----------

# The ALS is constructed by using the userIds to represent users, the movieIds to represent the items and the ratings column to represent the rating to fill.  Since the rating is not negative it, the non negative is True and preferences are not implicit. Also, a cold start strategy of drop is used to resolve the cold start problem (users with no reviews).

als = ALS(userCol="userId", itemCol="movieId", ratingCol="rating",
          nonnegative = True, implicitPrefs = False,
          coldStartStrategy = 'drop')

evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")

# COMMAND ----------

# MAGIC %md #### Tuning Hyperparameters using Cross Validation 

# COMMAND ----------

# By using k fold cross validation, we can compare ALS models using different hyper parameters to pick the ideal hyperparameters. For ALS, the parameters we will tune will be the rank & regParam which refers to the number of features to discover throughout the run.
# Grid
paramGrid = ParamGridBuilder() \
           .addGrid(als.rank, [10, 5, 1]) \
           .addGrid(als.regParam, [0.001,0.01,0.1])\
           .build()

# COMMAND ----------

# MAGIC %md #### Cross Validation and Best Model

# COMMAND ----------

cv = CrossValidator(
  estimator= als, estimatorParamMaps=paramGrid,
  evaluator=evaluator, numFolds=3)

# COMMAND ----------

# Running cross validation and choosing best model
cvmodel = cv.fit(train)
bestModel = cvmodel.bestModel

# COMMAND ----------

# Rank and regParam of model
print("rank: ", bestModel.rank, "\nregParam: ", bestModel._java_obj.parent().getRegParam())

# COMMAND ----------

# MAGIC %md #### Best Model Results

# COMMAND ----------

predictions = bestModel.transform(test)
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))

# COMMAND ----------

display(predictions)

# COMMAND ----------

# MAGIC %md The model seems to quite well here considering the sparsity of the matrix, at least according to the root mean square error. However, perhaps the algorithm can be improved by feeding it's communities of users instead of the whole users datasets, where users of said communities are likely to have similar prefereneces and hence predicting ratings would be easier. This will be done using Girwan Newman in the remaining tasks.

# COMMAND ----------

# MAGIC %md # TASK 3

# COMMAND ----------

# Loading libraries
import pandas as pd
import pyspark.sql.functions as f
from pyspark.sql.types import *
from pyspark.sql import Row
from operator import add
from collections import Counter

# COMMAND ----------

# Taking 27000 users from the dataset
sampleUsers1 = ratings.select("userId").distinct().sample(fraction=0.2)
list = sampleUsers1.take(27000)
sampleUsers = spark.createDataFrame(list,['userId'])
sampleUsers.count()

# COMMAND ----------

# creating ratings_small file for parquet
from pyspark.sql.functions import when
ratings_small = ratings[ratings["userId"].isin(sampleUsers.toPandas()["userId"].tolist())]
ratings_small.count()

# COMMAND ----------

#dbutils.fs.rm('small-rating.parquet', True) 

# COMMAND ----------

# saving ratings_small as parquet file. This won't work if file already exists.
ratings_small.write.parquet("ratings-small.parquet")

# COMMAND ----------

# MAGIC %md # TASK 4

# COMMAND ----------

# dropping 'timestamp' & 'rating' as they are not necessary for analysis for rest of the project.
ratings_small = ratings_small.drop('timestamp')
ratings_small = ratings_small.drop('rating')

# COMMAND ----------

# MAGIC %md #### Dataframes

# COMMAND ----------

# Two dataframes will be used to help divide the task up, each representing one of the user ids in the edge
ratings_small = spark.read.parquet("ratings-small.parquet")
df1 = ratings_small.select("movieId", "userId")
df2 = ratings_small.select("movieId", "userId").withColumnRenamed("movieId", "movieId2").withColumnRenamed("userId", "userId2")
UserMap = df1.join(df2, df1.movieId == df2.movieId2)

# COMMAND ----------

UserMap = UserMap.select("userId", "userId2")
UserMap.show(5)

# COMMAND ----------

UserMap.count()

# COMMAND ----------

# MAGIC %md #### Edges

# COMMAND ----------

# finding duplicate edges
from pyspark.sql.functions import col, sum
edgesWithDuplicates = UserMap.groupBy(UserMap.columns).count().filter(UserMap.userId != UserMap.userId2)
edgesWithDuplicates.orderBy("count", ascending=False).show(5)

# COMMAND ----------

# total number of duplicate edges
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

# filtering the edges which are greater than average
edgesFiltered = edges.filter("count>15")
edgesFiltered.count()

# COMMAND ----------

# MAGIC %md #### Vertices

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

# MAGIC %md #### Graph

# COMMAND ----------

# graph
graph = GraphFrame(verticesFiltered, edgesFiltered)
display(graph.edges)

# COMMAND ----------

# MAGIC %md # TASK 5

# COMMAND ----------

sc.setCheckpointDir("dbfs:/tmp/group1/checkpoint")
# Computes the connected components of the graph.
connectedComp = graph.connectedComponents()  # DataFrame with new vertices column “component”
connectedComp.count()
display(connectedComp.sort("component", ascending = False))

# COMMAND ----------

# MAGIC %md #### Subgraph

# COMMAND ----------

# finding max component for subgraph
max_component = connectedComp.filter(connectedComp["component"]==3)
edges_max_component_src = edges.join(max_component,max_component.id == 
                          edgesFiltered.src,'inner').select(edgesFiltered.src,edgesFiltered.dst)
edges_max_component = edges_max_component_src.join(max_component,max_component.id == 
                          edges_max_component_src.dst,'inner').select(edgesFiltered.src,edgesFiltered.dst)
display(edges_max_component)

# COMMAND ----------

edges_max_component.write.mode('overwrite').parquet("edges_max_component.parquet")

# COMMAND ----------

test = spark.read.parquet("edges_max_component.parquet")
display(test)

# COMMAND ----------

g2 = (connectedComp.describe("component").filter("summary = 'max'").select("component").collect()[0].asDict()['component'])

# COMMAND ----------

# MAGIC %md # TASK 6

# COMMAND ----------

import networkx as nx
import pandas as pd
import numpy as np

# COMMAND ----------

# set the number of top_k
top_k = 3

# COMMAND ----------

# MAGIC %md #### Sample graph for testing GN implementation.

# COMMAND ----------

vertices = sqlContext.createDataFrame([
  ("a", "Alice", 34),
  ("b", "Bob", 36),
  ("c", "Charlie", 30),
  ("d", "David", 29),
  ("e", "Esther", 32),
  ("f", "Fanny", 36),
  ("g", "Gabby", 60)], ["id", "name", "age"])

# COMMAND ----------

edges = sqlContext.createDataFrame([
  ("a", "b", "friend"),
  ("b", "a", "friend"),
  ("a", "c", "friend"),  
  ("c", "a", "friend"),
  ("b", "c", "friend"),
  ("c", "b", "friend"),
  ("b", "d", "friend"),
  ("d", "b", "friend"),
  ("d", "e", "friend"),
  ("e", "d", "friend"),
  ("d", "g", "friend"),
  ("g", "d", "friend"),
  ("e", "f", "friend"),
  ("f", "e", "friend"),
  ("g", "f", "friend"),
  ("f", "g", "friend"),
  ("d", "f", "friend"),
  ("f", "d", "friend")
], ["src", "dst", "relationship"])

# COMMAND ----------

src = edges.toPandas()["src"].tolist()
dst = edges.toPandas()["dst"].tolist()
edges_new = pd.DataFrame()
edges_new['src'] = src
edges_new['dst'] = dst
edges_new[['src','dst']]
G=nx.from_pandas_edgelist(edges_new,'src','dst')

# COMMAND ----------

# list all the nodes of the graph
G.number_of_nodes()
G.nodes

# COMMAND ----------

def _single_source_shortest_path_basic(G,s):
    S = [] # S is a dictionary. It stores the visited nodes
    P = {} # A dictionary to store parent node
    for v in G:
        P[v] = []
    sigma = dict.fromkeys(G, 0.0)    # sigma[v]=0 for v in G
                                    # set label for each node with 0 
    D = {}  #store the shortest path for each node to the begin node 
    sigma[s] = 1.0 # starting from vertex s and set the label of 1
    D[s] = 0 #initialise all the node to the begin node is 0
    Q = [s] # stack
    while Q:   # use BFS to find shortest paths
        v = Q.pop(0)
        S.append(v)
        Dv = D[v]
        sigmav = sigma[v]
        for w in G[v]:
            if w not in D:
                Q.append(w)
                D[w] = Dv + 1
#                 print("D[w] = ",w,D[w])
            if D[w] == Dv + 1:   # this is a shortest path, count paths
                sigma[w] += sigmav
#                 print("sigma[w] = ",w,sigma[w])
                P[w].append(v)  # predecessors
#     print (S)
#     print(P)
#     print(sigma)
    return S, P, sigma 

# COMMAND ----------

def betweenness(G):
    betweenness = dict.fromkeys(G, 0.0)  # b[v]=0 for v in G
    # b[e]=0 for e in G.edges()
    betweenness.update(dict.fromkeys(G.edges(), 0.0))
    for s in G:
        # single source shortest paths
        # use BFS
        S, P, sigma = _single_source_shortest_path_basic(G,s) # step 1 from lecture note
        # accumulation
        betweenness = _accumulate_edges(betweenness, S, P, sigma, s) # step 2 and 3 from lecture note
    # rescaling
    for n in G:  # remove nodes to only return edges
        del betweenness[n]
    betweenness = _rescale_e(betweenness, len(G))
    return betweenness

# COMMAND ----------

#step 2
def _accumulate_edges(betweenness, S, P, sigma, s):
    delta = dict.fromkeys(S, 0)
    while S:
        w = S.pop() # Poping value from last
        coeff = (1 + delta[w]) / sigma[w] # calculating coefficient value by using weight of edge
        for v in P[w]:
            c = sigma[v] * coeff
            # set order of the node from bottom to top
            if (v, w) not in betweenness:
                betweenness[(w, v)] += c
            else:
                betweenness[(v, w)] += c
            delta[v] += c
        if w != s:
            betweenness[w] += delta[w]
    return betweenness

# COMMAND ----------

# rescaling
#Finally, the true betweenness is obtained by dividing the result by two, since every shortest path will be discovered twice, once for each of its endpoints.
def _rescale_e(betweenness, n):
    scale = 0.5
    for v in betweenness:
         betweenness[v] *= scale
    return betweenness

# COMMAND ----------

# compute the edge betweenness

def CmtyGirvanNewmanStep(G):
    init_ncomp = nx.number_connected_components(G)    #number of components
    ncomp = init_ncomp
    while ncomp <= init_ncomp:
        #return a dict betweenness as the value
#         bw = nx.edge_betweenness(G, weight='weight')    #edge betweenness for G 
        bw = betweenness(G)
        print("current betweenness of each edge")
        print(bw)
        #find the edge with max centrality
        max_ = max(bw.values())
        #find the edge with the highest centrality and remove all of them if there is more than one!
        for k, v in bw.items():
            if float(v) == max_:
                G.remove_edge(k[0],k[1])    #remove the central edge
        ncomp = nx.number_connected_components(G)    #recalculate the no of components

# COMMAND ----------

# This method runs GirvanNewman algorithm and list communities as output after removing the top-K edges
def runGirvanNewman(G,k):
    while k>0:    
        CmtyGirvanNewmanStep(G)
        comps = list(nx.connected_components(G)) 
        print("communities:")
        print(comps)
        k = k-1
        if G.number_of_edges() == 0:
            break

# COMMAND ----------

def main():
    runGirvanNewman(G,top_k)
if __name__ == "__main__":
      main()

# COMMAND ----------

# MAGIC %md ##Girvan-Newman algorithm for subgraph

# COMMAND ----------

#create networkx graph
subG = spark.read.parquet("edges_max_component.parquet")
sub_src = subG.toPandas()["src"].tolist()
sub_dst = subG.toPandas()["dst"].tolist()
edges_sub = pd.DataFrame()
edges_sub['src'] = sub_src
edges_sub['dst'] = sub_dst
edges_sub[['src','dst']]
sub_G=nx.from_pandas_edgelist(edges_sub,'src','dst')

# COMMAND ----------

runGirvanNewman(sub_G,top_k)
