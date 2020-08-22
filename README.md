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
- **Deployment Framework :** Flask 1.1.0

## Project Outline 
- Data Preparation
- Data Summary
- Model Development
- Model Evaluation


## Data Collection and Cleaning
The taken dataset is a multiclass imbalanced dataset and consists of around **`0.1 Million`** news articles. 

![](/images/data_distribution.PNG)


## Data Summary 

Data summary includes the information about number of documents, words and unique words have in each category class. Also, include the length distribution of the  news artices in the dataset.

| ![accident](/images/accident.PNG) | ![crime](/images/crime.PNG) | ![economics](/images/economics.PNG) | ![sports](/images/sports.PNG) |![politics](/images/politics.PNG) |![entertainment](/images/entertainment.PNG) |

![length distribution](/images/len_dist.PNG)

**From this graphical information we can select the suitable  length of articles that we have to use for making every articles into a same length.**


## Model Development 

The used model architecture consists of a **embedding layer(`input_length = 300, embedding_dim = 128`), Conv layer(`128 , 5x5`), two bilstm layer(`nunits = 64`), two dense layer (`n_units = 28, 14`), and a softmax layer**. The Architecture looks like- 

![model](/images/model_architecture.PNG)

## Model Evaluation 

In this devloped model we have got **`84%`** validation accuracy which is not bad for such an imbalanced dataset. Besides, Confusion Matrix and other evaluation measures have been taken to determine the effectiveness of the developed model. From the confusion matrix it is observed that the maximum number of misclassified headlines are fall in the caltegory of **`Education, Art and Politics `** and it make sense because this categories news are kind of similar in words. The accuracy, precision, recall and f1-score result also demonstrate this issue. 

![training](/images/training_accuract.PNG)

![confusion](/images/confusion.PNG)

![performance](/images/performance_table.PNG)

**In conclusion, we have achieved a good accuracy of `84%` on this simple hybrid neural network for Bengali document categorization task. This accuray can be further improved by doing hyperparameter tunning and by employing more shophisticated network architecture with a large dataset.**


## Model Deployment

Here is the developed Flask App : [Document Categorizer App](https://bangla-document-categorization.herokuapp.com/)


![app](/images/app_interface.PNG)
