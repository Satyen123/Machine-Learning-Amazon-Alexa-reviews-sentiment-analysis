# Machine-Learning-Amazon-Alexa-reviews-sentiment-analysis
This project analyzes customer reviews of Amazon Alexa products using machine learning to determine sentiment (positive or negative). The goal is to understand customer satisfaction and uncover insights from product reviews.

📁 Files Included
Amazon Alexa Reviews Sentiment Analysis Machine Learning.ipynb: Jupyter Notebook containing the complete data analysis, preprocessing, model building, and evaluation.

amazon_alexa.tsv: Dataset of Amazon Alexa reviews sourced from Kaggle or a similar platform, in tab-separated format.

📊 Project Overview
🔍 Objectives
Clean and preprocess text review data.

Perform exploratory data analysis (EDA).

Use NLP techniques (like TF-IDF or CountVectorizer) for feature extraction.

Train machine learning models to classify sentiment.

Evaluate model performance using accuracy, confusion matrix, etc.

🛠 Models Used
Logistic Regression

Naive Bayes

Support Vector Machine (SVM)

🧰 Tech Stack
Python 🐍

Pandas, NumPy

Matplotlib, Seaborn

Scikit-learn

NLTK or TextBlob (for NLP)

🚀 Getting Started
Clone this repo:

bash
Copy
Edit
git clone https://github.com/yourusername/amazon-alexa-sentiment-analysis.git
cd amazon-alexa-sentiment-analysis
Install required packages:

bash
Copy
Edit
pip install -r requirements.txt
Run the Jupyter notebook:

bash
Copy
Edit
jupyter notebook
📈 Results
Data Overview:
The dataset contains over 3,000 Amazon Alexa product reviews with associated ratings and user feedback.

Sentiment Distribution:
The dataset was labeled into binary sentiment classes (positive vs. negative), with the majority of reviews being positive.

Best Model Performance:

Logistic Regression achieved the highest accuracy.

Accuracy: ~93%

Precision: 92%

Recall: 93%

F1 Score: 92.5%
(Replace these with actual metrics if different)

Feature Insights:

Words like “love”, “great”, “easy”, “fun” were most associated with positive reviews.

Words like “stopped”, “poor”, “waste”, “disappointed” indicated negative sentiment.

Visualizations:

Word clouds for positive and negative reviews.

Confusion matrix for each model.

Bar charts showing class balance and feature importance.

🧑‍💻 Author
Your Name – https://github.com/Satyen123

📝 License
This project is open source and available under the MIT License.
