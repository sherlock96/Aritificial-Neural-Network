# Churn Prediction in Banking using Aritificial-Neural-Network

Churn prediction is knowing whether a customer is likely to stop using service/product of the company in the near future.

The dataset we will work on contains data of 10,000 customers labeled with 1 and 0 denoting whether the customer churned out of the banking services. I have divided the training and test set in the ratio 80:20.

## Part 1 - Data Preprocessing                                                                                                             
Dataset is loaded and made into numpy array.                                                                                               Then categorical variables are encoded using LabelEncoders and OneHotEncoder                                                                                                                                                                                                      
## Part 2 - Making the ANN!                                                                                                               
Sequential Classifier is created using **keras** and input, ouput and hidden layers are added.                                             Then the classfier is compiled and trained.                                                                                                
## Part 3 - Making predictions and evaluating the model                                                                                   
Made predictions using confusion_matrix and classification_matrix
