## Bengali Document Categorization Using ConV-LSTM Net: Project Overview

- Created a tool that can categorizes the Bengali news articles into 12 diffferent categories (**`Art, Politics, International, Sports, Science, Economics, Crime, Accident, Education, Entertainment, Environment, Opinion`**) using **Deep Learning**.
- A publicly available [dataset](https://data.mendeley.com/datasets/xp92jxr8wn/2) of **`0.1 Million`** news articles is used to develop the system. The dataset consist 12 different categories news articles.      
- **`Word embeeding`** feature represtations technique is used for extracting the semantic meaning of the words.
- A deep learning model has been built by using a **`Convolutional Neural Network and Long Short Term Memory`**.
- The model performance is evaluated using various evaluation measures such as **`confusion matrix, accuracy , precision, recall and f1-score`**.
- Finally, developed a client facing API using **flask** and **heroku**.

## Resources Used
- **Developement Envioronment :** Google Colab
- **Python Version :** 3.7
- **Framework and Packages :** Tensorflow 2.1.0 , Scikit-Learn, Pandas, Numpy, Matplotlib, Seaborn
-**Deployment Framework :** Flask 1.1.0

## Project Outline 
- Data Preparation
- Data Summary
- Data Preparation for Model Building
- Model Development
- Model Evaluation


## Data Collection and Cleaning
The taken dataset is a multiclass imbalanced dataset and consists of around **`0.1 Million`** news articles. 

![](/images/data_distribution.PNG)


## Data Summary 

Data summary includes the information about number of documents, words and unique words have in each category class. Also, include the length distribution of the  news artices in the dataset.

| ![national](/images/national.PNG) | ![international](/images/international.PNG) | ![politics](/images/politics.PNG) | ![sports](/images/sports.PNG) |![amusement](/images/amusement.PNG) |![it](/images/it.PNG) |

![length distribution](/images/len_dist.PNG)

**From this graphical information we can select the suitable  length of headlines that we have to use for making every headlines into a same length.**


## Data Preparation for Model Building

The text data are represented by a encoded sequence where the sequences are the vector of index number of the contains words in each headlines. The categories are also encoded into numeric values. After preparing the headlines and labels it looks as -
![ecoded_sequence](/images/padded.PNG)  ![labels](/images/encoded_labels.PNG)

For Model Evaluation the encoded headlines are splitted into **Train-Test-Validation Set**. The distribution has -

![split](/images/train_test_split.PNG)


## Model Development 

The used model architecture consists of a **embedding layer(`input_length = 21, embedding_dim = 64`), GRU layer(`n_units = 64`), two dense layer (`n_units = 24, 6`), a dropout  and a softmax layer**. The Architecture looks like- 

![model](/images/model_architecture.PNG)

## Model Evaluation 

In this simple model we have got **`81%`** validation accuracy which is not bad for such an multiclass imbalanced dataset. Besides Confusion Matrix and other evaluation measures have been taken to determine the effectiveness of the developed model. From the confusion matrix it is observed that the maximum number of misclassified headlines are fall in the caltegory of **`Natinal, International and Politics `** and it make sense because this categories headlines are kind of similar in words. The accuracy, precision, recall and f1-score result also demonstrate this issue. 

![confusion](/images/confusion.PNG)

![performance](/images/performance.PNG)

**In conclusion, we have achieved a good accuracy of `84%` on this simple recurrent neural network for Bengali news headline categorization task. This accuray can be further improved by doing hyperparameter tunning and by employing more shophisticated network architecture with a large dataset.**





## Model Deployment

Here is the Flask App : [Document Categorizer App](https://bangla-document-categorization.herokuapp.com/)
