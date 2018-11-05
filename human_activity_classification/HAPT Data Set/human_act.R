#######################################Code for Human Activity Recognition Project#####################################
##1. Data is downloaded from UCI Machine Learning Repository:                                                        ##
##   http://archive.ics.uci.edu/ml/datasets/Smartphone-Based+Recognition+of+Human+Activities+and+Postural+Transitions##
##2. A Youtube video related to the data set: https://www.youtube.com/watch?v=XOEN9W05_4A                            ##
##3. Aim: classify six activities of a person using sensor data (561 attributes):                                    ##
##   WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING                                        ##
##4. In data folder "HAPT Data Set", only use "Train" and "Test" for modeling                                        ##
##5. Change working directory to the data folder "HAPT Data Set" before run this code                                ##
##6. For final results, run Part I and Part IV only.                                                                 ##
#######################################################################################################################

###################################################Part I: read data###################################################
train=read.table('./Train/X_train.txt', colClasses='numeric')
subj=as.integer(readLines('Train/subject_id_train.txt'))
trainy=as.integer(readLines('Train/y_train.txt'))
test=read.table('Test/X_test.txt', colClasses='numeric')
testy=as.integer(readLines('Test/y_test.txt'))
feature=as.character(unlist(read.table('features.txt')))

#select the six activities
train=train[trainy%in%1:6, ]
subj=subj[trainy%in%1:6]
trainy=trainy[trainy%in%1:6]
test=test[testy%in%1:6, ]
testy=testy[testy%in%1:6]
table(trainy) #how many obs for each activity
table(subj, trainy) #how many obs for each subject and each activity
#table(testy)
#######################################################################################################################

##################################################Part II: explore#####################################################
##subject effect (conclusion: very strong!)
#for each variable and each activity, perform an avona to test subject effect
pvalue=matrix(0, 6, 561)
for (i in 1:6) {
 for (j in 1:561) {
  pvalue[i, j]=anova(lm(train[trainy==i, j]~as.factor(subj[trainy==i])))[[5]][1]
 }
}
lpvalue=log(pvalue)
lpvalue[lpvalue==-Inf]=min(lpvalue[lpvalue!=-Inf])-10
image(1:6, 1:561, lpvalue, col=heat.colors(128)) #subject effect for each variable on each activity
mean(pvalue<1e-5) #overall strength of subject effect
rowSums(pvalue<1e-5) #for each activity
table(colSums(pvalue<1e-5)) #for each variable
feature[colSums(pvalue>0.01)>=5] #variables not influenced by subject effect
feature[colSums(pvalue[4:6, ]<1e-90)==3] #large subject effect in static activities

##two-way ANOVA (compare activity effect and subject effect)
#perform a two-way ANOVA on each variable
act_vs_subj<-pact<-numeric(561)
for (i in 1:561) {
 fit=anova(lm(train[, i]~as.factor(subj)*as.factor(trainy)))
 act_vs_subj[i]=fit[[2]][2]/fit[[2]][1]
 pact[i]=fit[[5]][2]
}
temp=which(act_vs_subj>50)[1] #relative great activity effect
interaction.plot(subj, trainy, train[, temp], type='l', fixed=T, col=rainbow(6, end=0.8), lty=1)
temp=which(act_vs_subj<1)[1] #relative small activity effect
interaction.plot(subj, trainy, train[, temp], type='l', fixed=T, col=rainbow(6, end=0.8), lty=1)
feature[which(act_vs_subj<1 & pact<1e-5)] #variables may cause trouble (try to avoid)
#two-way ANOVA on each variable (activity 1-3)
pact1to3=sapply(1:561, function(i) anova(lm(train[, i]~as.factor(subj)*as.factor(trainy), subset=which(trainy%in%1:3)))[[5]][2])
sum(pact1to3<1e-90) #how many useful variables for separating 1-3
temp=which(pact1to3<1e-90)[1]
interaction.plot(subj, trainy, train[, temp], type='l', fixed=T, col=rainbow(6, end=0.8), lty=1)
#two-way ANOVA on each variable (activity 4-6)
pact4to6=sapply(1:561, function(i) anova(lm(train[, i]~as.factor(subj)*as.factor(trainy), subset=which(trainy%in%4:6)))[[5]][2])
sum(pact4to6<1e-90) #how many useful variables for separating 4-6
temp=which(pact4to6<1e-90)[1]
interaction.plot(subj, trainy, train[, temp], type='l', fixed=T, col=rainbow(6, end=0.8), lty=1)
feature[pact4to6<1e-90] #useful variables for separating 4-6

##PCA
#PCA is for dimension reduction
pc1=prcomp(train)
cumvar=cumsum(pc1[[1]]^2)/sum(pc1[[1]]^2) #cumulative variance explained by PCs
plot(cumvar, type='b', cex=0.5) #scree plot
k1=which.min(pc1[[1]]^2>mean(pc1[[1]]^2))-1 #Kaiser's rule
trainpc1=pc1[[5]][, 1:k1]
pairs(trainpc1[, 1:3], col=rainbow(6, end=0.8)[trainy], cex=0.7)
#perform a scaled PCA
pc2=prcomp(train, scale=T)
cumvar=cumsum(pc2[[1]]^2)/561
plot(cumvar, type='b', cex=0.5)
k2=which.min(pc2[[1]]^2>1)-1
trainpc2=pc2[[5]][, 1:k2]
pairs(trainpc2[, 1:3], col=rainbow(6, end=0.8)[trainy], cex=0.7)
#Factor Analysis
fload1=pc1[[2]]%*%diag(pc1[[1]])[, 1:k1]
fload2=pc2[[2]]%*%diag(pc2[[1]])[, 1:k2]
pairs(fload1[, 1:4])
pairs(fload2[, 1:4])
#variables weight mostly in each factor
fload1[which.min(fload1)]=-max(fload1)
image(1:5, 1:561, t(fload1[, 1:5]), col=bluered(64))
feature[abs(fload1[, 1])>0.4] #important variables in the first factor
temp=which(act_vs_subj<1 & pact<1e-5)
sum(temp%in%which(pact<1e-300))/length(temp) #"trouble" variables in the first factor
temp=which(abs(fload1[, 1])>0.1)
sum(temp%in%which(pact<1e-300))/length(temp) #significant variables in the first factor
feature[abs(fload1[, 2])>0.15] #important variables in the second factor
temp=which(abs(fload1[, 2])>0.1)
sum(temp%in%which(pact1to3<1e-300))/length(temp) #significant dynamic variables in the second factor
feature[abs(fload1[, 3])>0.1] #important variables in the third factor
temp=which(abs(fload1[, 3])>0.1)
sum(temp%in%which(pact4to6<1e-50))/length(temp) #significant static variables in the third factor
#######################################################################################################################

##################################################Part III: modeling###################################################
#"subcv" is a function to perform cross validation based on subjects
#only for classification problems
library(e1071)
library(randomForest)
library(nnet)
subcv=function(data, label, subj, ntest=3, method='SVM', ...) {
 data=as.data.frame(data)
 label=as.factor(label)
 s=sample(unique(subj))
 nfold=length(s)%/%ntest
 if (nfold==0) stop('ntest must be smaller than subject level!')
 res=list(nfold)
 for (i in 1:nfold) {
  testid=subj%in%s[(ntest*i-ntest+1):(ntest*i)]
  x=data[!testid, ]
  y=label[!testid]
  xtest=data[testid, ]
  ytest=label[testid]
  if (method=='SVM') fit=svm(y~., x, ...)
  else if (method=='RF') fit=randomForest(x, y, ntree=300, ...)
  else if (method=='GLM') fit=multinom(y~., x, ...)
  else stop('please choose a correct method!')
  pred=predict(fit, xtest)
  res[[i]]=cbind(ytest, pred)
 }
 res=do.call('rbind', res)
 tt=table(res[, 1], res[, 2])
 data.matrix(cbind(tt, class.error=1-diag(tt)/rowSums(tt)))
}
#svm
system.time(tt1<-subcv(pc1[[5]][, 1:100], trainy, subj))
1-sum(diag(tt1[, -7]))/sum(tt1[, -7])
system.time(tt2<-subcv(train, trainy, subj, gamma=1/700))
1-sum(diag(tt2[, -7]))/sum(tt2[, -7])
#random forest
system.time(tt1<-subcv(pc1[[5]][, 1:100], trainy, subj, method='RF', mtry=3))
1-sum(diag(tt1[, -7]))/sum(tt1[, -7])
system.time(tt2<-subcv(train, trainy, subj, method='RF', mtry=7))
1-sum(diag(tt2[, -7]))/sum(tt2[, -7])
#glm
system.time(tt1<-subcv(pc1[[5]][, 1:100], trainy, subj, method='GLM'))
1-sum(diag(tt1[, -7]))/sum(tt1[, -7])
#######################################################################################################################

###################################################Part IV: testing####################################################
pc=prcomp(train)
k=100
trainpc=pc[[5]][, 1:k]
testpc=predict(pc, test)[, 1:k]
#svm
library(e1071)
system.time(SVM<-svm(trainpc, as.factor(trainy), gamma=1/150))
con_svm=table(testy, predict(SVM, testpc))
con_svm=data.matrix(cbind(con_svm, class.error=1-diag(con_svm)/rowSums(con_svm)))
1-sum(diag(con_svm[, -7]))/sum(con_svm[, -7])
#randomForest
library(randomForest)
system.time(RF<-randomForest(train, as.factor(trainy), test, as.factor(testy), ntree=300, mtry=7))
plot(RF) #check convergence
feature[head(order(-RF[[9]]), 20)] #important features
con_rf=RF[[17]][[3]]
1-sum(diag(con_rf[, -7]))/sum(con_rf[, -7])
#glm
library(nnet)
system.time(GLM<-multinom(trainy~., as.data.frame(trainpc)))
con_glm=table(testy, predict(GLM, as.data.frame(testpc)))
con_glm=data.matrix(cbind(con_glm, class.error=1-diag(con_glm)/rowSums(con_glm)))
1-sum(diag(con_glm[, -7]))/sum(con_glm[, -7])
#knn
library(class)
system.time(con_knn<-table(testy, knn(trainpc, testpc, trainy, 32)))
con_knn=data.matrix(cbind(con_knn, class.error=1-diag(con_knn)/rowSums(con_knn)))
1-sum(diag(con_knn[, -7]))/sum(con_knn[, -7])
#######################################################################################################################