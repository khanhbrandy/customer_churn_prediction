# Customer Churn Analysis and Prediction
This project works on a dataset containing data on the customers of a call center.

## Dataset description
In addition to basic demographic information of customers, such as state, area, and their mobile/telephone/customer service usage, the dataset also includes rather or not a particular customer churned, which then will be utilized to develop a Churn Prediction Model. 
The main aim of this project is to discover major patterns in the behavior of churned and non-churned (regular) customers, then use them to early identify high-risk customers. Such a churn prediction model allows the company to take proactive actions to keep existing customers from churning and then provide them with better support and promotions or to gauge what needs might not be being met. 

## Model development
Detailed report is provided in the <b><a href="model_development.ipynb"> model_development.ipynb</a></b> file. 

## Concluding remarks
Through a series of data processing and machine learning techniques, a fairly accurate Churn prediction model was developed and ready for production. On top of the predictive model, many actionable insights have been found along the way. The most important finding is that Customers total usage of day minutes (total_day_minutes), Number of customer service calls (number_customer_service_calls), and Customer’s international calling plan (international_plan) are among the most important attributes to determine whether a customer would churn or not. This makes more sense in the real business contexts when we are all agree that users tend to churn when they have bad customer experience (and might have to make many customer service calls to complain). Therefore, in addition to improve the quality of day time calls and offer impressive international plans, the company should place more emphasis on taking care of those who make more customer service calls than usual to ensure that their needs and difficulties are satisfied and supported timely.  

## Deployment
In order to make the model more production-ready, an API has been developed using (Python) Flask, a web development framework. The source code of this part is saved in the Deployment folder. 

<img src="Deployment\demo.png" alt="Demo API" />