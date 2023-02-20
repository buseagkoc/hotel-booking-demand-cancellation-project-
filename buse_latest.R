#install.packages("countrycode")
library(ISLR)
library(ggplot2)
library(tidyverse)
library(dplyr)
library(DataExplorer)
library(caret)
library(rpart)
library(rpart.plot)
library(randomForest)
library(regclass)
library(corrplot)
library(countrycode)
library(rattle)
library(rpart.plot)
library(RColorBrewer)
library(pROC)
library(knitr)

# Data Import
Hotel=read.csv("~/Desktop/DM/hotel_bookings.csv",header=T, stringsAsFactors=TRUE)

# Data Structure 
head(Hotel)
dim(Hotel) 
str(Hotel)
glimpse(Hotel)

# Summary Statistics 
summary(Hotel)


# Clean missing values
is.na(Hotel)
plot_missing(Hotel)
sum(is.na(Hotel))

n <- length(Hotel$children)
for (i in 1:n) {
  if (is.na(Hotel$children[i]))
    Hotel$children[i] <- 0}


#Replacing undefined at meal column as SC .Both means no meal package
Hotel$meal <-replace(Hotel$meal,Hotel$meal=='Undefined','SC')
Hotel$meal <- factor(Hotel$meal)
unique(Hotel$meal)
Hotel$meal


# Dropping some columns with high missing values (agent, company, and reservarion status date)
Hotel <- Hotel[,-c(24,25,32)]

# obtain a correlation matrix
cor_matrix(Hotel)
#Plotting the correlation matrix
dfnum = dplyr:: select_if(Hotel, is.numeric)
dfnum = data.frame(lapply(dfnum, function(x) as.numeric(as.character(x))))
res=cor(dfnum)
corrplot(res, method="color", type="upper", tl.col="black" )
# all_correlations (to get the pairwise correlations between every variable.)
all_correlations(Hotel,sorted="significance")
# Looks like lead_time&arrival_date_week_number have the strong association ->Make a scatterplot of their relationship
plot(lead_time ~ arrival_date_week_number,data=Hotel)


#Converting the other variables into factor
Hotel<-Hotel%>%
  mutate(
    hotel=as.factor(hotel),      
    is_canceled=as.factor(is_canceled),
    country=as.factor(country),
    market_segment=as.factor(market_segment),
    distribution_channel=as.factor(distribution_channel),
    is_repeated_guest=as.factor(is_repeated_guest),
    reserved_room_type=as.factor(reserved_room_type),
    assigned_room_type=as.factor(assigned_room_type),
    deposit_type=as.factor(deposit_type),
    customer_type=as.factor(customer_type),
    reservation_status=as.factor(reservation_status),
    arrival_date_day_of_month=as.factor(arrival_date_day_of_month),
    arrival_date_month=as.factor(arrival_date_month),
    arrival_date_year=as.factor(arrival_date_year)
    
  )

str(Hotel)

#Creating two new columns to calculate total number of days stayed and total cost

Hotel <- Hotel %>% 
  mutate(stay_nights_total = stays_in_weekend_nights + stays_in_week_nights,
         stay_cost_total = adr * stay_nights_total)

summary(Hotel$stay_nights_total)
summary(Hotel$stay_cost_total)

# Checking for outliers &  updating that value with the mean of adr
Hotel%>%
  filter(adr>1000)

Hotel = Hotel%>%
  mutate(adr = replace(adr, adr>1000, mean(adr)))


### --------- Exploration ----------- ###

#Exploring the number of countries involved (from where the most guests are coming?)
Hotel%>%
  group_by(country)%>%
  summarise(num=n())%>%
  arrange(desc(num))

#Continents
Hotel$continent <- Hotel$country %>% countrycode(origin = "iso3c",destination = "continent") 
Hotel$continent[is.na(Hotel$continent)]<- "Other"
Hotel$continent %>% as.factor()
Hotel %>% mutate(is_canceled = as.factor(is_canceled)) %>%  ggplot(aes(x = continent, fill = is_canceled)) +
  scale_fill_discrete(
    name = "Booking Status",
    breaks = c("0", "1"),
    label = c("Cancelled", "Not Cancelled")
  )+ geom_bar(position = "fill") +scale_y_continuous(labels = scales::percent)
 +geom_text(aes( label = scales::percent(..prop..),
                y= ..prop.. ))
 
# Checking Number of arrival Month (when is the busy month?)

d <- Hotel %>% 
  group_by(arrival_date_month) %>%
  count() %>%
  arrange(match(arrival_date_month,month.name))
d
barplot(table(Hotel$arrival_date_month),xlab="Number of People",main="When is the Busy Month?", xlim=c(0,14000), cex.names = 0.7, cex.axis = 0.7,horiz=T, las=1)

#Plotting the Number of arrival Month of cancellations
Hotel %>%
  mutate(arrival_date_month = factor(arrival_date_month,
                                     levels = month.name
  )) %>%
  count(hotel, arrival_date_month, is_canceled) %>%
  group_by(hotel, is_canceled) %>%
  mutate(proportion = n / sum(n)) %>%
  ggplot(aes(arrival_date_month, proportion, fill = is_canceled)) +
  geom_col(position = "dodge") +
  scale_y_continuous(labels = scales::percent_format()) +
  facet_wrap(~hotel, nrow = 2) +
  labs(
    x = NULL,
    y = "Percentage of Booking Status ",
    fill = NULL
  ) +scale_fill_discrete(
    name = "Booking Status",
    breaks = c("0", "1"),
    label = c("Cancelled", "Not Cancelled")
  ) 

# Booking status by month
ggplot(Hotel, aes(arrival_date_month, fill = factor(is_canceled))) +
  geom_bar() + geom_text(stat = "count", aes(label = ..count..), hjust = 1) +
  coord_flip() + scale_fill_discrete(
    name = "Booking Status",
    breaks = c("0", "1"),
    label = c("Cancelled", "Not Cancelled")
  ) +
  labs(title = "Booking Status by Month",
       x = "Month",
       y = "Count") + theme_bw()

#Year of arrival
barplot(table(Hotel$arrival_date_year))

# Checking preferred hotel type
barplot(table(Hotel$hotel))

# Checking Total stay nights (usually, how long staying the hotel?)
barplot(table(Hotel$stay_nights_total))


#NUmber of city hotel and Resort Hotel cancelled or not cancelled (City v.s. Resort)
ggplot(Hotel, aes(x= is_canceled,  group=hotel)) + 
  geom_bar(aes(y = ..prop.., fill = factor(..x..)), stat="count") +
  geom_text(aes( label = scales::percent(..prop..),
                 y= ..prop.. ), stat= "count", vjust = -.5) +
  facet_grid(~hotel) +
  labs(y = "Percent", fill="is_canceled") +
  scale_fill_discrete(
    name = "Cancellation",
    breaks = c("2", "1"),
    label = c("Cancelled", "Not Cancelled")
  )+
  scale_x_discrete("Canceled",labels = c("No","Yes"))+
  scale_y_continuous(labels = scales::percent)

#City Hotel Cancellation is more
Hotel %>% group_by(Hotel$hotel)  %>% summarise(length(is_canceled))

resort_hotel <- Hotel[which(Hotel$reservation_status!="No-Show"& Hotel$hotel == "Resort Hotel"),]
city_hotel <- Hotel[which(Hotel$reservation_status!="No-Show"&Hotel$hotel == "City Hotel"),]

#stay duration for both the hotels
p <-ggplot(data=Hotel, aes(stay_nights_total))+geom_density(col="red")+facet_wrap(~hotel)+theme_bw()+xlim(c(0, 20)) 
p                                               
p+ geom_vline(aes(xintercept=mean(stay_nights_total)),
              color="blue", linetype="dashed", size=1)


### --------- Model Building ----------- ###

#Based on correlational matrix and the nature of our large dataset, we selected specific variables
Hotel = Hotel %>% select(is_canceled, hotel,adults,children,babies,meal,stay_nights_total,stay_cost_total, lead_time, arrival_date_month, arrival_date_day_of_month, stays_in_weekend_nights, stays_in_week_nights, meal, market_segment, distribution_channel, is_repeated_guest, previous_cancellations, previous_bookings_not_canceled, reserved_room_type, booking_changes, deposit_type, days_in_waiting_list, customer_type, adr, required_car_parking_spaces, total_of_special_requests)
dim(Hotel)

#Checking imbalance of the data
ggplot(Hotel, aes(x=is_canceled)) +
  geom_bar(aes(y = (..count..)/sum(..count..))) +scale_x_discrete(
    name = "Cancellation Status",
    breaks = c("0", "1"),
    label = c("Cancelled", "Not Cancelled")
  ) +scale_y_continuous(labels = scales::percent)+
  labs(title = "Class Imbalance",
       x = "Cancellation",
       y = "Percentage") + theme_bw()

#Even though there is imbalance in the data,it is quite mild so we can ignore it for now

#Splitting dataset 
set.seed(123)
split <- sample(c(TRUE,FALSE),nrow(Hotel),prob = c(0.80,0.20),replace = TRUE)
train <- Hotel[split, ]
test <- Hotel[!split, ]
summary(train)
summary(test)
dim(train)
dim(test)

#10th Fold Cross Validation split
#Randomly shuffle the data
set.seed(123)
Hotel <- Hotel[sample(nrow(Hotel)),]
folds <- cut(seq(1,nrow(Hotel)),breaks=10,labels=FALSE)
for(i in 1:10){
  testIndexes <- which(folds==i,arr.ind=TRUE)
  testcv <- Hotel[testIndexes, ]
  traincv <- Hotel[-testIndexes, ]
}

#Logistic Regression 
set.seed(123)
model1 <- glm(is_canceled ~.,
              data=train , family = "binomial")

summary(model1)

prob=predict(model1,test,type="response")
prob1=rep(0,nrow(test))
prob1[prob>0.5]=1
cmlr= confusionMatrix(as.factor(prob1),mode="everything",test$is_canceled) 
cmlr

#Logistic regression (CV)
set.seed(123)
glm.model <- glm(is_canceled ~.,
                 data=traincv , family = "binomial")

summary(glm.model)

prob=predict(glm.model,testcv,type="response")
prob1=rep(0,nrow(testcv))
prob1[prob>0.5]=1
cmlr= confusionMatrix(as.factor(prob1),mode="everything",testcv$is_canceled) 
cmlr


#Decision Tree

# Converting some variables to dummy variables
Hotel$previous_cancellation2<-ifelse(Hotel$previous_cancellations<1,0,1)
Hotel$leadtime2 <-ifelse(Hotel$lead_time<=14,0, 1)
Hotel$booking_changes2<-ifelse(Hotel$booking_changes==0,0, 1)

set.seed(123)   
dtree <-
  rpart(
    is_canceled ~.,
    data = train,
    method = "class",
    control=rpart.control(cp=0, maxdepth = 3))

summary(dtree)
printcp(dtree)
plotcp(dtree)

min_xerror<-dtree$cptable[which.min(dtree$cptable[,"xerror"]),]
min_xerror

minx_tree<-prune(dtree, cp=min_xerror[1])
rpart.plot(minx_tree)

predict <- predict(dtree, test, type = "class")
confusionMatrix(predict,mode="everything",test$is_canceled)

# Decision Tree Model (CV)
set.seed(123)   
dtreecv <-
  rpart(
    is_canceled ~.,
    data = traincv,
    method = "class",
    control=rpart.control(cp=0, maxdepth = 17))

summary(dtreecv)
printcp(dtreecv)
plotcp(dtreecv)

min_xerrorcv<-dtreecv$cptable[which.min(dtreecv$cptable[,"xerror"]),]
min_xerrorcv

minx_treecv<-prune(dtreecv, cp=min_xerrorcv[1])
rpart.plot(minx_treecv)

predict_dt <- predict(dtreecv, testcv, type = "class")
confusionMatrix(predict_dt,mode="everything",testcv$is_canceled)

#Random Forest
set.seed(123) 
rf_train<-randomForest(is_canceled~.,   
                       data=train,
                       sampsize=c(500, 500),#Increased
                       ntree=500,                     
                       cutoff=c(.5,.5), 
                       mtry=2,
                       importance=TRUE,
                       do.trace = 100)
importance(rf_train)

#Tuning the model
set.seed(123)              
tuning <- tuneRF(x = train%>%select(-is_canceled),
                 y = train$is_canceled,mtry=4, #Increased
                 sampsize=c(500, 500), #Increased
                 ntreeTry = 500)

#Best Model 
rf_best<-randomForest(is_canceled~.,         
                      data=train,         
                      ntree=500,                     
                      cutoff=c(.32,.68), #adjusted from .5,.5
                      mtry=8,
                      sampsize=c(16000,34000), #increased sampsize, take longer to compute but help with accuracies
                      importance=TRUE)
rf_best

predict <- predict(rf_best, test, type = "class")
confusionMatrix(predict, test$is_canceled)

#Best Model (CV)
rf_bestcv<-randomForest(is_canceled~.,         
                        data=traincv,         
                        ntree=500,                     
                        cutoff=c(.32,.68), #adjusted from .5,.5
                        mtry=8,
                        sampsize=c(16000,34000), #increased sampsize, take longer to compute but help with accuracies
                        importance=TRUE)
rf_bestcv
plot(rf_bestcv, main = "Error rate of random forest")
predictcvrf <- predict(rf_bestcv, testcv, type = "class")
confusionMatrix(predictcvrf, testcv$is_canceled)

#Comparison table (Cross Validation Values)

table = data.frame(Model <- c("Log", "DT", "RF"), Accuracy <- c(0.81,0.8236,0.8589), Precision <- c(0.7979,0.8415,0.8792), Recall <- c(0.9336,0.8857,0.8985), F1 <- c(0.8605,0.8630,0.8887))
kable(table, col.names = c("Model","Accuracy","Precision","Recall","F1 Score"))


#ROC Curves using cross validation (10th fold)

#log model
roc_log = roc(testcv$is_canceled, prob, plot=TRUE, print.auc=TRUE)
#dt model
testcv$tp= predict_dt 
roc_dt= roc(response= testcv$is_canceled, predictor = factor(testcv$tp, ordered=TRUE), plot=TRUE, print.auc=TRUE)
#rf model
testcv$predictcvrf= predictcvrf
roc_rf= roc(response= testcv$is_canceled, predictor = factor(testcv$predictcvrf, ordered=TRUE), plot=TRUE, print.auc=TRUE)

plot(roc_log,print.auc=TRUE,print.auc.y=.4, col="green")
plot(roc_dt,print.auc=TRUE,print.auc.y=.1,col="blue",add=TRUE)
plot(roc_rf,print.auc=TRUE,print.auc.y=.3, col="red",add=TRUE)
legend("bottomright", c("Log", "DT", "RF"), lty=1, 
       col = c("green", "blue","red"), bty="n", inset=c(0,0.15))
title(main = "ROC curve (CV)",line = 2.5)

