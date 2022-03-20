---
  title: "Assignment 4"
author: "Nemin Dholakia"
date: "11/6/2021"
output:
  html_document: default
pdf_document: default
---
  
  # Setting working directry to current File location
  ```{r}
Pharmaceuticals <- read.csv("Pharmaceuticals.csv")
setwd("D:/MSBA/Fundamentals of Machine Learning/ML assignment 4")
```

# Import required libraries

```{r}
library(tidyverse)  # data manipulation
library(cluster)    # clustering algorithms
library(factoextra) # clustering algorithms & visualization
```

# A: Kmeans Algo on Data set.
# Import the data set- Pharmaceuticals.csv

```{r}
library(readr)
Pharmaceuticals <- read_csv("Pharmaceuticals.csv", 
                            col_types = cols(Market_Cap = col_number(), 
                                             Beta = col_number(), PE_Ratio = col_number(), 
                                             ROE = col_number(), ROA = col_number(), 
                                             Asset_Turnover = col_number(), Leverage = col_number(), 
                                             Rev_Growth = col_number(), Net_Profit_Margin = col_number()))
data_df <- data.frame(Pharmaceuticals[,3:10])
```

# Will remove missing values from the dataset.

```{r}
data_df <- na.omit(data_df)
```

# Standardized all the values from column 3-9 (Market Cap - Net Profit Margin)

```{r}
data_df_scale <- scale(data_df)
```

###Now we are going to compute the distance between the rows of the data. We will use Euclidean distance which is the default distance used.But we also use any other distance
```{r}
distance <- get_dist(data_df_scale)
fviz_dist(distance, gradient = list(low = "#00AFBB", mid = "white", high = "#FC4E07"))
```

###Run K-Means Clustering Algo Now we will run K-means algo, with intially 2 clusters and with an mutilple intial configuration of 25.
```{r}
kmeans_model <- kmeans(data_df_scale, centers = 2, nstart = 25)
str(kmeans_model)
```

###There are several parameters that we get as an output.The important one's are:
#Clusters : The vector of integers which tells us the cluster to which each point is allocated.
#centers : Matrix of cluster centers
#Size : The size of each cluster (In this we have 11 values in C1 and 10 in C2)

# When we print the model itself, we see the actual grouping within the clusters.
```{r}
kmeans_model
```


# To view and understand this model, we plot this using fviz_cluster() method
```{r}
fviz_cluster(kmeans_model, data = data_df_scale)
```

###Since we know that K is a hyperparameter that the user needs to set, we will run several iterationd of Kmeans algo, each time changing the value of K.
#This will help us find the optimal value of K.
```{r}
kmeans_model3 <- kmeans(data_df_scale, centers = 3, nstart = 25)
kmeans_model4 <- kmeans(data_df_scale, centers = 4, nstart = 25)
kmeans_model5 <- kmeans(data_df_scale, centers = 5, nstart = 25)
kmeans_model6 <- kmeans(data_df_scale, centers = 6, nstart = 25)
kmeans_model7 <- kmeans(data_df_scale, centers = 7, nstart = 25)
```


#Plots for comparing
```{r}
n1 <- fviz_cluster(kmeans_model,  geom = "point",  data = data_df_scale) + ggtitle("k = 2")
n2 <- fviz_cluster(kmeans_model3, geom = "point",  data = data_df_scale) + ggtitle("k = 3")
n3 <- fviz_cluster(kmeans_model4, geom = "point",  data = data_df_scale) + ggtitle("k = 4")
n4 <- fviz_cluster(kmeans_model5, geom = "point",  data = data_df_scale) + ggtitle("k = 5")
n5 <- fviz_cluster(kmeans_model6, geom = "point",  data = data_df_scale) + ggtitle("k = 6")
n6 <- fviz_cluster(kmeans_model7, geom = "point",  data = data_df_scale) + ggtitle("k = 7")
library(gridExtra)
grid.arrange(n1, n2, n3, n4, n5, n6, nrow = 4)
```


###Now as the graph depicts, we find ourself in new type of situation.
#This situation is to decide which value of k to consider?
#Thankfully, we have 3 methods to determine this.
#These methods are:
1. Elbow method
2. Silhouette method
3. Gap statistic

###In this scenario we will use the Elbow method. We will consider the value of k, where there is bend in the plot.We use fviz_nbclust() method, which calculates this for us.
```{r}
set.seed(123)
fviz_nbclust(data_df_scale, kmeans, method = "wss")
```

Here we will not use k = 7 as the value goes up before going down.
So we will choose k = 5.

# Similarly we calculate Silhouette method and Gap statistic.

Silhouette method
This method determines how well an object lies in a cluster.
In this, a high average silhouette width indicates good clustering.

```{r}
fviz_nbclust(data_df_scale, kmeans, method = "silhouette")
```

###As we can see, the Silhouette method gives us 2 clusters.

#Now we can extract the final result for our kmeans method
#by setting value of k = 5 (For Euclidean Method)
```{r}
set.seed(123)
final_model <- kmeans(data_df_scale, 5, nstart = 25)
print(final_model)
```

###And we can visualize the result using fviz_cluster()
```{r}
fviz_cluster(final_model, data = data_df_scale)
```

#We can also take k = 2 for our final result and see what result we get by that (For Silhouette method).

```{r}
set.seed(123)
final_model2 <- kmeans(data_df_scale, 2, nstart = 25)
print(final_model2)
```
# And we can visualize the result using fviz_cluster()


```{r}
fviz_cluster(final_model2, data = data_df_scale)
```


###But in this case we will consider the value of k = 5.

# B. 
We get 5 clusters with following rows from data.
Cluster 1 : 2, 18 
Cluster 2 : 6, 8, 12 
Cluster 3 : 3, 7, 10, 16, 19, 21 
Cluster 4 : 11, 13, 15, 17 
Cluster 5 : 5, 9, 14, 20 

# C.  
Yes. There appears to be a pattern emerging when one observes the cluster formation
the values from the Median_recommendation column.
For Instance, cluster 3 that has highest Market Cap, ROA and asset turnover does not
have strong sell recommendation, instead it has 1 STRONG BUY and 4 HOLD recommendations.

# D. 
The Clusters can be loosely named as:
  Cluster 1. Moderate buy and hold cluster.
Cluster 2. Moderate buy and strong hold cluster (because of 2 Hold values in it)
Cluster 3. Solid hold and strong buy cluster
Cluster 4. Hold and moderate buy cluster(because of 2 entries for each of cluster.)
Cluster 5. Moderate buy and Sell Cluster.