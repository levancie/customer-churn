# Predicting Customer Churn with Machine Learning
by Leo Evancie

It’s [well established](https://www.optimove.com/resources/learning-center/customer-acquisition-vs-retention-costs) that customer acquisition is typically more costly than customer retention. This is of particular importance to companies who bill customers on a periodic basis. In order to keep revenue streams healthy, these companies need to do everything they can to keep their customers. But are all customers equally likely to “churn” (i.e., stop being customers)? Or are there clues in their profile and behavior that may help us predict, and hopefully prevent, their dropoff?

## 1. Data Wrangling
_Code notebook found [here](https://github.com/levancie/customer-churn/blob/main/notebooks/1-Data-Wrangling.ipynb)._

For this project, I used an IBM sample dataset (via [Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn)) consisting of over 7,000 customers from a fictional telco company. The data included basic demographic information (e.g., gender, senior citizen status), customer tenure, which of a wide range of services they opted into (e.g., multiple phone lines, paperless billing), monthly fee and payment method, total charges, and whether or not they churned.

![RawData](/images/RawData.png)

As is typical of Kaggle datasets, there were minimal data quality issues, requiring only the recasting of `TotalCharges` from object to float; filling some null values in that column with zero, since they corresponded to valid but brand-new customers; and recoding `SeniorCitizen` from 1/0 to 'Yes'/'No' for consistency with the other categorical features.

## 2. Exploratory Data Analysis
_Code notebook found [here](https://github.com/levancie/customer-churn/blob/main/notebooks/2-EDA.ipynb)._

### Target

I found that 26.54% of the ~7,000 customers had churned. This put the dataset somewhat out of balance, but I felt there were enough customers to still take a straightforward classification approach. Further work could include exploring corrections for imbalanced data sets, e.g., oversampling from the churned class when preparing the training data.

### Categorical Features

Most of the features were categorical in nature. I conducted bivariate analyses, comparing the churn percentage of customers in each category. A few standouts:

![SeniorCitizen](/images/SeniorCitizen.png)

Senior citizens were more likely to churn.

![InternetService](/images/InternetService.png)

Fiber optic users had a high churn rate, while that of customers with no internet service was low.

![OnlineSecurity](/images/OnlineSecurity.png)

Customers who opted into online security service showed higher rates of churn. This pattern held for several other related services, including OnlineBackup, PaperlessBilling, and TechSupport. I found this surprising, especially in the case of PaperlessBilling. Many of these services are usually considered to be conveniences demanded by engaged customers. As it turns out, though, while these services may be in demand, they may be associated with churn rather than retention.

![Contract](/images/Contract.png)

Customers billed month-to-month were far likelier to churn than those on 1- or 2-year contracts.

Other notable patterns were identified, meanwhile two features (`PhoneService` and `MultipleLines`) did not appear to show meaningful relationships to churn; those two features were discarded.

### Numerical Features

![Tenure](/images/Tenure.png)

While both churned and non-churned customers showed a clustering around zero months of tenure (i.e., closing their account within a month of opening it), the two groups otherwise show almost mirrored distributions. The clear bulk of non-churned customers had a tenure of around 70 months, reflecting the fact that most of the non-churned customers have been active for the entire time-scope of the data. Clearly, tenure can be a strong predictor of churn, especially after the first few months.

![MonthlyCharges](/images/MonthlyCharges.png)

We saw a somewhat higher distribution of monthly charges for churned customers. This is in line with the previous finding of high churn among month-to-month paying customers.

## 3. Preprocessing
_Code notebook found [here](https://github.com/levancie/customer-churn/blob/main/notebooks/3-Preprocessing.ipynb)._

Preparing the categorical features consisted of creating dummies with `pandas`' .get_dummies() function. Numerical features were rescaled with the `scikit-learn` StandardScaler. The Churn column, originally 'Yes'/'No', was re-encoded to 1/0 for classification. I executed the train/test split before rescaling the numerical columns to avoid 'leaking' data into the testing set; I fit the StandardScaler to the training data, then transformed both the training and test data.

## 4. Modeling
_Code notebook found [here](https://github.com/levancie/customer-churn/blob/main/notebooks/4-Modeling.ipynb)._

With training data accounting for 70% of my original 7,000 customers, I briefly tested three common classificaton algorithms with their default hyperparameters: LogisticRegression, RandomForestClassifier, and SupportVectorClassifier.

My metric of primary interest was recall. Increasing as false negatives decrease, recall captures how well the model avoids the problem of misclassifying churned customers as retained. In this business context, that would be the worse type of misclassification. It would be far less detrimental to classify a retained customer as churned; such would merely lead to additional outreach and care for customers who may not need it.

Logistic regression showed the most promising initial results, so I drilled down and tuned hyperparameters in the hope of improving recall and overall accuracy. Surprisingly, this had little effect, and actually slightly worsened model performance. As such, the overall best model was the baseline logistic regression. This is so surprising a result that it sheds light on the need to revisit the data preparation steps.

![ConfusionMatrix](/images/ConfusionMatrix.png)

![Metrics](/images/LR_metrics_test.png)

The model yielded nearly 80% accuracy, but misclassified nearly half of the churned customers.

## Conclusions

The bivariate analyses performed in the EDA highlighted some important patterns of which sales and marketing teams should take note. Customers who pay month-to-moth, senior citizens, those with fiber optic service, those who enroll in many of the convenience services like paperless billing, and those within their first few months of service are particularly likely to churn (and other factors shown in the EDA notebook). Each of these has implications for both attracting and servicing customers.

While the predictive model has room for improvement, it is a promising start. Even in its current state, it could guide some strategic and efficient customer interventions, pointing customer service teams in the direction of at least half of the customers at risk of churning, for a small fraction of the financial and time cost of applying extra retention efforts to all customers.

## Credits

Thank you to Mukesh Mithrakumar for your mentorship.

This is a capstone project for [Springboard's](https://www.springboard.com/) Data Science Career Track.

Leo Evancie, 2022.
