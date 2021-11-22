# Problem Statement
You are working as a data scientist with HR Department of a large insurance company focused on sales team attrition. Insurance sales teams help insurance companies generate new business by contacting potential customers and selling one or more types of insurance. The department generally sees high attrition and thus staffing becomes a crucial aspect.

To aid staffing, you are provided with the monthly information for a segment of employees for 2016 and 2017 and tasked to predict whether a current employee will be leaving the organization in the upcoming two quarters (01 Jan 2018 - 01 July 2018) or not, given:

Demographics of the employee (city, age, gender etc.)
Tenure information (joining date, Last Date)
Historical data regarding the performance of the employee (Quarterly rating, Monthly business acquired, designation, salary)

Private Leaderboard Rank - 6

# Approach
Following data is about an organization that is concerned about its staffing. Losing employees frequently impacts the morale of the organization and hiring new employees is more expensive than retaining existing ones. Some factors regarding to the monthly performance, quarterly ratings, salary etc. are given in this dataset so as to acknowledge a behavior pattern within the employees that affects them and eventually leads them to make a decision whether to leave the organization or not.

The very first step is that the target variable is not specifically mentioned in the train data. For constructing the target variable as shown in the definition, one should first look at the ‘LastWorkingDate’ column. Wherever the column has null values, that means the employee is continuing his/her work at the organization at least in the next year. Wherever any date record is appearing, that means the employee has left the organization on that particular date. So as per definition, we will put 0 where LWD column is null and 1 where LWD column has a date.

After that I have applied some data preprocessing techniques such as grouping, scaling, encoding etc. and try several supervised machine learning models. Whichever model gives the best accuracy as per the data we are feeding in the model, I have used that for predicting our test data.

The test data contains only the employee ids. Thus, for taking the performance and demographics of employees, I havd to perform inner join function between test and train to get those parameters in the same order as it is arranged in test data. After that we will predict the status of employees as per the performance parameters. At the end I have put that in the submission file and finally upload the solution.

# Data Preprocessing Steps
1) The very first step taken was to remove the date columns such as ‘Reporting Date’, 'DateofJoining' and 'LastWorkingDate' because thay are unique for each employee that will not be useful for predicting attrition.

2) There are multiple records for a particular employee with a particular employee id. Hence, I had to group the data on the basis of Employee ID.

3) Grouping all columns except ‘Total Business Value’ and will keep the last occurrence of the employee id in that particular group. The reason for this is that the ‘Salary’ and ‘Designation’ are changing as employee progresses in the given time period.

4) Grouping the ‘Total Business Value’ and will add them in each group of employee id as that is the performance given by the employee to the company.

5) Concatenating both these groups to get a new dataframe.

6) Checking distribution of data in each category for categorical columns using countplots whereas I have checked distribution of data in numerical columns using histograms. I have also checked the outliers in numerical columns using boxplots and also through a user-defined function. Outliers are ignored as the percentage of them present in the data is low.

7) I have also checked the Variance Inflation Factor of columns for multicollinearity within the columns as this would hinder the precision of prediction in the model.

8) Categorical columns those have text categories have to be encoded in dummy variables and make a column for each category that will help in efficient prediction of target.

9) Numerical columns have been scaled using MinMaxScaler() as the data in between the columns have too much variations. For example, age column has numbers ranging from 21 to 58 while salary column has numbers ranging from 10747 to 188418. Hence, they need to be brought in the same scale for efficient prediction.

10) Viewing the value counts of target variable, we can see that data in both the categories are imbalanced i.e., 765 zeros and 1616 ones which can create a problem in the recall of predicted data. Hence, I will upsample the zeros and make their count equal to ones category so the balanced data will help in better prediction.

# Deciding The Final Machine Learning Model
1) After all the data preprocessing steps, many classification models have been tried and tested after splitting the data into train and test. However, final submission code here contains only final model which has been decided as per the local accuracy and public leaderboard score.

2) Here, I have chosen the Support Vector Classifier model and went ahead for the final prediction. Hyperparameters such as kernel, regularization and gamma have been tuned so that maximum accuracy can be achieved

3) Support Vector Classifier model has many advantages compared to other machine learning models, one of them being the facility of soft margin. SVC will allow to misclassify some data records intentionally so as to adapt to any kind of data. The main motive of SVC is not to try to achieve the perfect accuracy but to gradually improve the accuracy.

4) In the final prediction, it is giving 467 zeros and 274 ones i.e. there are 467 employees mentioned in test data that are not likely to leave within the next 2 quarters of year 2018. Whereas there are 274 employees that are quite likely to leave in the same time period.

# Other Submission made
One of the other submissions made has also been uploaded here in this folder. The final machine learning model selected here was Ada-Boost Classifier model.

In this, categorical variables are directly Label Encoded. Also the data imbalance has been maintained due to inherent weight updation properties of Ada-Boost Classifier model of updating the weak learners after every iteration.

The final data has been trained using the AdaboostClassifier() which gave me a local accuracy of 0.80, public leaderboard score of 0.7126 and private leaderboard score of 0.7508.


# Final Submission F1_Score
F1_Score in Public Leaderboard - 0.7307
F1_Score in Private Leaderboard - 0.7373
