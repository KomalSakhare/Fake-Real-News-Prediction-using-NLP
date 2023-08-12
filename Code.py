------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Imported the required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# to ignore all warning messages that may arise during the execution
import warnings
warnings.filterwarnings("ignore")

# To load the dataset
df = pd.read_csv("news.csv")

# to display the first 5 rows of the dataset
df.head()

# To check the size of dataset, there are 6334 rows and 4 columns
df.shape

# To randomly display any 10 rows from the dataset, to get a view of how the data is in the dataset.
df.sample(10)

# Now, firstly i have seprated the column text based on the label value, i.e FAKE or REAL
# After seprating the text column, i have got two different sets out of which one contains the FAKE news and the other contains the REAL news
# Secondly, I have joined all the FAKE news text together using the space. 
# And also joined all the REAL news text together using the space.
# This will provide me with two different combined sets of news, where I can do the further processing.

real = " ".join(df[df["label"] == "REAL"]["text"])
real

fake = " ".join(df[df["label"] == "FAKE"]["text"])
fake

# Imported the class WordCloud from the wordcloud library.
from wordcloud import WordCloud

# Created the WORDCLOUD of both the combined text i.e REAL & FAKE.
# Creating the WORDCLOUD will help to understand the content of the news,
# WORDCLOUD displays the words used frequently in the FAKE or REAL news
# The Words having larger sizes is considerd to be the most used words, this helps to detect which words are used to differentiate the news between FAKE or REAL.

wc = WordCloud(width = 800, height = 800,
              background_color = "white",
              min_font_size = 10)
wc.generate(real)
plt.imshow(wc)
plt.axis('off')
plt.show

wc =WordCloud(width = 800, height = 800,
             background_color = "white",
              min_font_size = 10)
wc.generate(fake)
plt.imshow(wc)
plt.axis('off')
plt.show()

# Now to train our model to predict the news catgorize we need to train our model, and we require the specific libraries.
from sklearn.model_selection import train_test_split

# Now X is independent column, Y is the dependent column or (target) column.
# The X column is our text column, which will help to train our model
# The Y column is our target column, which will help to predict the news categorie 
# The remaining column does not add much value for prediction, hence not considered.
x = df["text"]
y = df["label"]

xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size = 0.2,random_state = 1)

# Imported the tfidf Vectorizer.
from sklearn.feature_extraction.text import TfidfVectorizer

# The tfidf Vectorizer particularly useful for converting text data into a format that machine learning algorithms can work with effectively.
#The TF-IDF vectorizer assigns numerical values to words in a document

# In this first we remove the STOP-WORDS from the documents, Stopwords include ("is", "the", "and")
# Next for training the model , fitted and transformed the training data, and transformed the testing data.
tv = TfidfVectorizer(stop_words = "english")
xtrain_tv = tv.fit_transform(xtrain)
xtest_tv = tv.transform(xtest)

xtest_tv

# First i have used the Decision Tree Classifier, for prediction, so imported it.
from sklearn.tree import DecisionTreeClassifier

# created the object for Decision Tree and then fitted the trainig model in it, to predict the target variable.
dt = DecisionTreeClassifier()
dt.fit(xtrain_tv,ytrain)

# After completion of the training, predicted the output i.e the target variable on the testing dataset.
ypred = dt.predict(xtest_tv)

# Imported the clssification report to check the accuracy
from sklearn.metrics import classification_report

# Printed the classification report, which has specified the overall accuracy of the model, also the accuracy of FAKE & REAL news
# The overall accuracy of model is 80%, the accuracy to predict only FAKE news is 81%, and to predict the REAL news is also 80%.
print(classification_report(ytest,ypred))

# Now to improve the accuracy of the model, I have used another algorithm i.e PASSIVE-AGGRESSIVE-CLASSIFIER.

from sklearn.linear_model import PassiveAggressiveClassifier

# As done for decision tree, same I have fitted the training data for Passive-Aggressive Classifier 
pac = PassiveAggressiveClassifier()
pac.fit(xtrain_tv,ytrain)

# Prediction of the testing data
ypred = pac.predict(xtest_tv)

# Printed the classification report.
print(classification_report(ytest,ypred))

# After using the Passive-Aggressive Classifier, I have got the accuracy of model as 94%.
# Also the accuracy for predicting the FAKE & REAL news is also 94%.

# A confusion matrix is a tabular representation that shows the count of correct and incorrect predictions made by classification model

# imported the confusion matrix
from sklearn.metrics import confusion_matrix

# to create the confusion matrix, we need to pass the value for prediction, so
# we need to pass the testing data and the predicted data of the targeted column.
confusion_matrix(ytest,ypred, labels = ['FAKE','REAL'])

# array([[611,  40], this it confusion matrix I got.
#       [ 36, 580]]
# this specifies that there are 611 true positive, 580 true negative and 36 false positive, 40 false negative.

# After completion of all the trainig to our model, now its time to test our model on unseen data
# we will test our model by giving some unknown data to our model to predict the value.

text1 = "Trump is coming to India to meet president "
text2 = "Political parties are travelling to different states for gathering votes"
text3 = "America is 90 % Republican Country"
text4 = "Trump is supporting BJP in this election"

def predict_text(text):
  txt = tv.transform([text])
  predict = pac.predict(txt)
  return (predict)

predict_text(text1)
predict_text(text2)
predict_text(text3)
predict_text(text4)



--------------------------------------------------------------------------------------------------------------------------------------------------------------------
