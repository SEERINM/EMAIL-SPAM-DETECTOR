
# 📧 Email Spam Detection using Machine Learning

This project is a simple yet powerful **Email Spam Detection System** built with **Python, Scikit-learn, and Streamlit**.  
It uses **Natural Language Processing (NLP)** techniques and the **Naive Bayes classifier** to detect whether a given message is **Spam** or **Not Spam**.



## 🚀 How to Run the Project

### 1. **Clone the Repository**

git clone [https://github.com/your-username/email-spam-detector.git](https://github.com/SEERINM/EMAIL-SPAM-DETECTOR.git)
cd email-spam-detector


### 2. **Install Dependencies**

Make sure you have Python installed. Then install the required libraries:


pip install -r Requirements.txt


If you don’t have `requirements.txt`, you can install manually:


pip install pandas numpy scikit-learn streamlit

### 3. **Download the Dataset**

Make sure `spam.csv` is in the project directory. You can download it from:
[https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

### 4. **Run the App**

streamlit run SpamDetection.py

## ✅ Features

* Simple and intuitive web interface using Streamlit
* Text preprocessing using CountVectorizer
* Machine Learning model using Multinomial Naive Bayes
* Real-time spam prediction


## 🧪 Sample Messages to Test

### ✅ Not Spam

* `Let's catch up tomorrow after class.`
* `Please find the attached invoice.`
* `Happy Birthday! Have a great day.`

### ❌ Spam

* `You've won a $1000 Amazon gift card! Click here now.`
* `URGENT: Your account has been blocked.`
* `Win a trip to Paris! Enter your details.`

## 🛠️ Technologies Used

* Python
* Pandas, Numpy
* Scikit-learn
* Streamlit
* Natural Language Processing



