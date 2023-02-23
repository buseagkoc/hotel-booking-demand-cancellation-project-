# hotel-booking-demand-cancellation-project-
Data Mining (ML) project for classifying hotel booking demand/cancellations. 


Model Building

After cleaning and exploring the dataset, our team used 3 classification algorithms that would be
suitable for predicting cancellations in our analysis. 
Based on the correlational matrix and the nature of our large dataset, and also the Random Forest algorithm
didn’t run if there is a variable that has more than 53 levels, we removed the country variable (178 variables).
We decided to remove the variable instead of creating dummy variables to ignore the dummy trap. Lastly, as
seen in Fig 5, our target variable in our dataset has a skewed class proportion, which means that our data is
imbalanced. However, since the degree of imbalance is mild. We didn’t take any action in terms of class
imbalance. For the further analyses, we could improve our models’ accuracies by mitigating the imbalance
problem.

<img width="449" alt="Screen Shot 2023-02-20 at 8 09 14 AM" src="https://user-images.githubusercontent.com/117880976/220117667-898ccd2c-64dd-4398-b582-657c1176d67d.png">

After choosing our final variables, we splitted the data into one training set containing a random
sample of 80% of observations, and one testing set with the remaining 20%. Using the training data, our team
has developed a logistic regression model for cancelation dummy (‘is_canceled’) with 26 variables as
predictors. Our team also conducted a 10th Fold Cross Validation split to Randomly shuffle the data. Cross
Validation is used to measure the test error related to a model to evaluate its performance. In order to compare
our models, resampling is an effective step that estimates the error of a model on unobserved data, helps to
determine the flexibility of the model, and provides an effective parameter selection. We also pay attention to
use the same seed value to make sure that we get the same output for randomization. In all classification
models, we improved the accuracy results of models with and without cross-validation by tuning their
hyperparameters (Please see the R code).

Model Comparison
According to the results of logistic regression, decision tree, and random forest, we have created a
comparison table with cross validated accuracy, precision, recall and F1 scores of each model. Besides each
model’s performance metrics, we also graph the ROC curve (Fig 6) using cross validation (10th fold) to
compare our models more clearly. In the ROC curve the true positive rate shows sensitivity and it is plotted in
funcion of false positive rates that shows specificity for different points. The area underneath (AUC) presents
an overall estimate of performance across all possible classification thresholds. This is a better way than
overall accuracy since it is not based on one specific cutpoint like overall accuracy. Logistic regression model
seems to have the best ROC AUC result in our analysis. According to our cross validated Logistic Regression
model only 1771 bookings were misclassified as canceled and only 497 cancellations were misclassified as
booking.
The result of the comparison table (Table 1) showed that the random forest model has the best
performance in accuracy, precision, and F1 score compared to logistic regression and decision tree model. The
decision tree model has the second best performance in accuracy, precision, and F1 score. However, the
logistic regression model has the best recall compared to the decision tree and random forest model. Thus, the
random forest model is the best model in this comparison. According to our Random Forest model’s confusion
matrix results, only 924 bookings were misclassified as canceled and only 762 cancellations were misclassified
as booking.

<img width="559" alt="Screen Shot 2023-02-20 at 8 09 59 AM" src="https://user-images.githubusercontent.com/117880976/220117876-b01ae8cd-677b-4193-8f3f-fdcca24c2ec8.png">

<img width="544" alt="Screen Shot 2023-02-20 at 8 10 28 AM" src="https://user-images.githubusercontent.com/117880976/220117961-139ae208-9eaa-4219-bf0d-9dc9238ec71d.png">


