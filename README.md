# Amazon Review Sentiment Analysis

This project focuses on performing sentiment analysis on Amazon customer reviews using various NLP and machine learning techniques.



## 📦 Installation
Before running the project, make sure you have all the required Python libraries installed.  
You can install them all at once by running:

```bash
pip install -r requirements.txt
```

You can check all the required Python libraries inside `requirements.txt` file





## 📂 Project Structure

A recommended project layout:

```text
amazon-review-sentiment/
│
├── data/                 # datasets (train/test) – not included in repo
├── notebooks/            # Jupyter notebooks for exploration & experiments
├── src/                  # source code for preprocessing, training, evaluation
│   └── script.py         # main script to run the project
│
├── requirements.txt      # list of dependencies
└── README.md             # project documentation
```


## 🚀 How to Run

The project is executed from the command line and requires 4 input arguments:

1- train_data → path to the training dataset

2- test_data → path to the testing dataset

3- model → the model to use for sentiment analysis. Options include:

	lr → Logistic Regression
	svm → Support Vector Machine
	rf → Random Forest
	gb → Gradient Boosting
	nb → Naive Bayes
	lstm → LSTM Neural Network

4- output_name → name of the output file (default: results.csv)

Example:

```bash
python script.py data/train.csv data/test.csv lr results/result
```


This command trains a **Logistic Regression** model on `train.csv`, evaluates it on `test.csv`, and saves the predictions to `results` folder.


