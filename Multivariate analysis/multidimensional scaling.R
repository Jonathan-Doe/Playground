#utf-8
setwd('/home/johndoe/repos/Playground/Multivariate analysis/')
library("factoextra")
library("CCA")
library("cluster")
library("fossil")
library("MASS")
library("gdata")
library("dplyr")
library("tidyr")
# Read data 
data <- data.frame(read.table("Сlustering with known #0f clusters data/mult7.txt"))


#Clustering
km <- kmeans(data, centers = 6)
pal <- rainbow(6)


#Calculate distance matrix
d <- dist(x = data, method = "euclidean")

#Multidimensional scaling and vizualization
mds <- cmdscale(d, k=2, eig=TRUE)
x <- mds$points[,1]
y <- mds$points[,2]
plot(x, y,col=pal[km$cluster],cex=0.2)
