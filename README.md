# Fake-Real-News-Prediction-using-NLP
In this project, we have developed a machine learning model for classifying news articles as either fake or real using Natural Language Processing techniques, to help users distinguish between trustworthy and potentially misleading news sources.

I have utilized a dataset of labeled news articles, and through a combination of text preprocessing, feature extraction, and model training, I have created an accurate predictive model.

Key Features:

--> Preprocessing : Tokenization, stop-word removal, to prepare text data for analysis.
--> TF-IDF Vectorization: Conversion of text into numerical vectors using TF-IDF weighting.
--> Model Selection : Experimentation with various classification algorithms to determine the best-performing model.
--> Descision Tree Classifier : To predict the news , FAKE or REAL
--> Passive Aggressive Classifier : To predict the news , FAKE or REAL, to make the model more accurate.
--> Evaluation : Utilization of precision, recall, F1-score, and confusion matrix to assess model performance.



TF-IDF (Term Frequency-Inverse Document Frequency) is a widely used technique in natural language processing and information retrieval to represent text documents as numerical vectors.
It's particularly useful for converting text data into a format that machine learning algorithms can work with effectively. The TF-IDF vectorizer assigns numerical values to words in a document, capturing their importance within that document relative to a collection of documents.

Term Frequency (TF): This measures the frequency of a term (word) within a document. It's calculated as the number of times a term appears in a document divided by the total number of terms in that document.

Inverse Document Frequency (IDF): This measures the rarity of a term across all documents in a collection. It's calculated as the logarithm of the total number of documents divided by the number of documents containing the term.




A Decision Tree Classifier is a machine learning algorithm used for both classification and regression tasks. It works by recursively partitioning the feature space into subsets based on the values of input features. Each partition is associated with a specific class label (in the case of classification) or a predicted value (in the case of regression)

Decision Tree Classifiers have several advantages, including interpretability, the ability to handle both numerical and categorical data, and their effectiveness in capturing complex relationships in the data. However, they can easily overfit the training data if not properly tuned or regularized.



The Passive-Aggressive Classifier is a machine learning algorithm used for binary classification tasks. It's particularly well-suited for online learning scenarios and situations where data arrives in a stream. The algorithm aims to make correct predictions while minimizing the increase in the magnitude of the model's weights

The name "Passive-Aggressive" reflects the algorithm's behavior:

"Passive" updates occur when the model predicts correctly and doesn't need to adjust its weights.
"Aggressive" updates happen when the model predicts incorrectly, and it adjusts its weights aggressively to correct the mistake.



A confusion matrix is a tabular representation that shows the count of correct and incorrect predictions made by a classification model, organized by the actual and predicted classes. It helps assess the model's performance by revealing the true positives, true negatives, false positives, and false negatives.
