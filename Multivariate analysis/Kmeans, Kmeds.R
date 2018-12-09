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
data <- read.table("Сlustering with known #0f clusters data/mult7.txt")

######################## K means ##################### 
#WSS
fviz_nbclust(data, kmeans, method = "wss",k.max = 20)


#Silhoutte
fviz_nbclust(data, kmeans, method = "silhouette",k.max = 20)
# k = 6


# Vizualization on 2 principal comps
km <- kmeans(data, 6, nstart = 20)
pal <-c ("black","red","blue","green","magenta","chocolate")
plot(princomp(data)$scores[,1:2],col=pal[km$cluster],cex=0.2)


# Vizualization on canonical comps
k <- length(levels(as.factor(km$cluster)))
n <- nrow(data)
C <- matrix(data=as.numeric(rep(km$cluster,k) == rep(1:k, each=n)),ncol=k, nrow=n)
cc_res <- rcc(data, C, 0.1, 0.1)
plot(cc_res$scores$xscores[,3:4],col=pal[km$cluster],cex=0.2, xlab = '3rd comp', ylab = '4th comp')

########################## K medoids ########################

#WSS
fviz_nbclust(data, pam, method = "wss",k.max = 20)


#Silhoutte
fviz_nbclust(data, pam, method = "silhouette",k.max = 20)

# Vizualization on 2 principal comps
kp <- pam(data, 6,metric = 'euclidean')
pal <-c ("black","red","blue","green","magenta","chocolate")
plot(princomp(data)$scores[,1:2],col=pal[kp$cluster],cex=0.2)


# Vizualization on canonical comps
k <- length(levels(as.factor(kp$cluster)))
n <- nrow(data)
C <- matrix(data=as.numeric(rep(kp$cluster,k) == rep(1:k, each=n)),ncol=k, nrow=n)
cc_res <- rcc(data, C, 0.1, 0.1)
plot(cc_res$scores$xscores[,3:4],col=pal[kp$cluster],cex=0.2, xlab = '3rd comp', ylab = '4th comp')


#Calcualte Rends index
rand.index(kp$clustering, km$cluster)
table(kp$clustering, km$cluster)


############# External Dataset - HTRU2 (https://archive.ics.uci.edu/ml/datasets/HTRU2)
data <- data.frame(read.csv("HTRU2/HTRU_2.csv",header = FALSE)) %>% drop_na()

n <- 1000
sampled_data <- sample_n(data, n, replace = FALSE)

fviz_nbclust(sampled_data, kmeans, method = "wss",k.max = 10)


#Silhoutte0
fviz_nbclust(data, kmaens, method = "silhouette",k.max = 20)
