# Projects
This is **Susmitha Yanamadala**, **Data Science Masters Grad with 4+ years of experience**. This repository consists of projects implemented as part of the course work and self-study.

**Project List and Details:**

**SQL & Python:**

1. **IMDB Data Management and Analysis**: Project done as part of EAS 503 Python Programming course work.
   * Created a SQLite database and applied third normal form to the tables. Using the sqlite3 library in Python, the tables were successfully created and 10000000 records of data is loaded into the database.
   * Subsequently, SQL queries were employed in Python to retrieve the data from the database for exploratory data analysis, allowing for the discovery of valuable insights.

 Skills Used: Python (pandas, sqlite3, matplotlib, seaborn, plotly), SQL, Database Management Concepts (Normalization), Exploratory Data Analysis.

2. **Bank Marketing Campaign Data Management and Web Integration**: Project done as part of CSE 560 Data Models Query Language course work.
   * Created a PostgreSQL database and decomposed tables to BCNF. The tables were created and data is loaded into the database using the psycopg2 library in Python.
   * Extensive testing of the database and data analysis was conducted, involving various commands such as insert, update, delete, select, group by, order by, having and where, as well as advanced concepts like joins, subqueries, recursive queries, common table expressions (CTEs)/temporary tables, and triggers.
   * Query execution analysis was carried out, and the cost of query execution was reduced by optimizing them through index creation.
   * A real-time website was developed to enable data retrieval directly from the database.

 Skills Used: Python (pandas, psycopg2, flask), PostgreSQL, Database Management Concepts (Normalization, CTE, Index Creation, Triggers, Query Optimization), HTML.

**Supervised & Unsupervised Machine Learning:**

1. **Heart Disease Prediction**: Project done as part of EAS 508 Statistical data Mining-I course work
   * Vital signs, including blood pressure, heart rate, cholesterol, and blood sugar, along with factors such as age, gender, food habits, and lifestyle, are utilized to predict the likelihood of a person being diagnosed with heart disease. Conducted exploratory data analysis to comprehend the data.
   * Implemented a spectrum of classification models: Logistic Regression, Decision Tree, Random Forest, K Nearest  Neighbor, Gaussian Naive Bayes, and Support Vector Classifier and achieved 88% accuracy using Random Forest model.

 Skills Used: Python (numpy, pandas, matplotlib, seaborn, sklearn), Data Cleaning, Data pre-processing, Data Augmentation using Kmeans Clustering, Exploratory Data Analysis, Feature Engineering, Classification (Logistic Regression, Decision Tree, Random Forest, K Nearest  Neighbor, Gaussian Naive Bayes, and Support Vector Classifier)
     
2. **Regression Analysis on Penguin & Flight Price Datasets**: Project done as part of CSE 574 Intro to ML course work.
   * Spearheaded a machine learning project, crafting Logistic, Linear, Ridge Regression, and Ridge Regression with Gradient Descent models from scratch for the Penguins and Flight Price Prediction datasets, bypassing traditional frameworks like Scikit-learn.
   * Executed rigorous data visualization and cleaning on both datasets, unveiling pivotal insights and ensuring data integrity.
   * Deployed Logistic Regression on the Penguins dataset to forecast species, emphasizing hyperparameter tuning to refine model accuracy.
   * Leveraged the Flight Price Prediction dataset with Linear Regression, Ridge Regression, and Ridge Regression using Gradient Descent, delving into ticket price predictions while addressing model sensitivities and harnessing L2 regularization for enhanced generalization and robustness.

 Skills Used: Python (numpy, pandas, matplotlib, seaborn, pickle), Data Cleaning, Data pre-processing, Exploratory Data Analysis, Feature Selection, Logistic Regression with sigmoid cost function and gradient descent from scratch, Regression Analysis (Linear Regression, Ridge Regression, Ridge Regression with Gradient Descent) from scratch.
     
3. **Movie Clustering for Recommender Systems**: Project done as part of EAS 509 Statistical Data Mining-II course work.
   * Data cleaning/pre-processing and analysis were conducted, focusing on features such as popularity, vote average, vote count, and release year and month for clustering purposes.
   * Both K-means and hierarchical clustering algorithms were implemented. The outcomes of the clustering process can be leveraged to enhance Movie Recommendation Systems.

 Skills Used: R (lubridate, corrplot, ggplot2, ggally, factoextra, tidyverse, dplyr, tibbel, readr, cluster, plotly), R Markdown, Data Cleaning, Data pre-processing, Exploratory Data Analysis, Feature Selection, clustering (K Means, Hierarchical).

4. **Credit Card Default Prediction**: Personal project
   * Constructed a predictive model employing machine learning techniques to forecast credit card defaults, attaining an F1 score of 0.92 with a Random Forest classifier.
   * Leveraged SMOTE technique to address data imbalance. Explored a variety of classification algorithms such as Logistic Regression, Decision Trees, and Support Vector Machines (SVM) to optimize performance.

 Skills Used: Python (numpy, pandas, matplotlib, seaborn, sklearn), Data Cleaning, Data pre-processing, Data Augmentation using SMOTE technique, Exploratory Data Analysis, Feature Engineering, Classification (Logistic Regression, Decision Tree, Random Forest, and Support Vector Classifier).
  
**Deep Learning:**

1. **Binary Classification using Neural Network** (Pytorch):
   * Built a neural network model for binary classification using a dataset with 7 input features and 766 samples.
   * Handled data preprocessing including data type conversion, normalization using StandardScaler, and feature engineering based on heatmap analysis.
   * Designed a three-layer architecture: Input layer with 7 nodes; hidden layer with 128 nodes; output layer with 2 nodes, achieving an accuracy of 80.9%.
   * Optimized the model using dropout regularization, varied optimizer algorithms (Adam, Adamax, Adagrad), and diverse activation functions (Tanh, LeakyReLU, ELU).
   * Applied learning rate scheduler, batch normalization, weight decay, and K-fold cross-validation to enhance model robustness.
     
2. **CIFAR-100 Image Classification using AlexNet** (Pytorch):
   * Worked on a dataset of 30,000 images (64x64) resized to 224x224, distributed across three categories: dogs, food, and vehicles.
   * Implemented the AlexNet architecture with a batch size of 50, running for 20 epochs, utilizing cross-entropy loss and the Adam optimizer.
   * Achieved a notable accuracy of 96.1% on testing data, 94.2% on training, and 93.9% on validation sets.
   * Made architectural modifications by removing a layer and shifting to SGD optimizer, which resulted in a more stable model performance with 92% training, 93.1% validation, and 92.7% test accuracies.
   * Ensured model resilience with early stopping and model checkpointing mechanisms.
     
3. **Street View House Numbers Image Classification** (Pytorch):
   * Handled a dataset containing street view house numbers images (73257 training and 26032 testing) of size 32x32, resized to 224x224.
   * Utilized a modified AlexNet architecture from prior work with a batch size of 512 over 5 epochs and the Adam optimizer.
   * Used data augmentation techniques like random horizontal flipping and random cropping to improve the model's generalization to new, unseen data; particularly noted significant improvement with the horizontal flip and faced challenges with random cropping due to significant information loss.
   * Managed to achieve a consistent performance across training, validation, and testing phases with 85.7%, 85.7%, and 85% accuracies respectively.

**Reinforcement Learning:**

1. **Multi Agent Reinforcement Learning For Fire Rescue** (Pytorch and Gymnasium):
   * Implemented a decentralized Multi-Agent Environment where agents work to rescue a person from fire. Employed tabular methods (Q-learning, SARSA) and deep RL methods (DQN, DDQN) in a custom grid world environment, analyzing and comparing their performance.
   * Extended the evaluation to an existing collaborative MARL -  OpenAI particle environment (Simple Reference from Petting Zoo MPE) using DQN and DDQN,to assess their effectiveness in the context of the established environment.
   
2. **Stock Trading Environment with Q-learning Implementation** (Pytorch):
   * Developed a Q-learning-based reinforcement learning model, harnessing GPU computational power to optimize trading strategies in a simulated stock trading environment. Achieved a cumulative profit increase of 20% over baseline strategies.
   * Evaluated and fine-tuned various parameter settings and learning rates to enhance the agent's decision-making capabilities and trading performance.

**NLP & Generative AI:**

1. **Mapping Clinical Trials with ICD-11 Using BERT**:
   *  Developed a multi-label classification model leveraging BERT to automatically tag over 50,000 clinical trial descriptions with WHO disease codes.
   * Implemented advanced text preprocessing, tokenization, and fine-tuned the model using PyTorch and Hugging Face Transformers, achieving 92% accuracy.

2. **Online Patient Conversation Classifier**:
   * Developed an NLP and ML-based system to classify online patient conversations, utilizing methods like Bag of Words, TF-IDF vectorization, and logistic regression for classification. The process included preprocessing text data, feature extraction, model training, and evaluation.
   * Achieved a precision and F1 score of 0.932, indicating the model's accuracy in distinguishing patient conversations. Enhanced classification accuracy through the application of feature engineering techniques.

**Computer Vision:**

1. **Car Damage Detection System** (OpenCV and Pytorch):
   * Engineered and processed a dataset of 4000 images with 9000 annotations for a car damage detection system leveraging advanced computer vision techniques including Cascade R-CNN, SSD, and YOLOv4, attaining a mean average precision of 85%.

**Time-Series:**

1. **Demand Forecasting For Optimizing Inventory Planning**:
   * Designed demand forecasting models for unique combinations of customers and items in Azure ML Studio, ensuring pre-processing of sales data.
   *Applied a range of time series forecasting methods such as ARIMA, SARIMA, Exponential Smoothing, Seasonal Decomposition (STL), and SARIMAX, validating each model's accuracy via MAPE.

**Recommender Systems:**

1. **Travel Destination Recommendation System**:
   * Developed a real-time travel suggestion platform with Python, harnessing the User-Based Collaborative Filtering algorithm for personalized destination suggestions.
   * Refined the system through comprehensive A/B testing, ensuring users received precise and relevant travel recommendations.

**Data Mining:**

1. **Market Basket Analysis** (Databricks and Pyspark):
   * Consolidated, refined, and integrated 40 CSV datasets—ranging from transactional to inventory data—for a hardware retail store.
   * Extracted actionable insights on customer behaviors and purchasing trends using Python. Executed the FP-Growth algorithm on Azure DataBricks with PySpark, segmenting both products and customers.
   * Achieved swift implementation, wrapping up the project in a 2-week timeframe. Assessed solution efficacy via A/B testing to ensure actionable and accurate results.

**Azure:**

1. **Azure Data Platform Implementations for B2B**:
   * Engineered end-to-end Azure-based solutions for integrating data from various ERP systems into Azure improving data accessibility.
   * Configured BYOD export jobs and integrated CDM exports into Azure SQL and Synapse’s serverless databases respectively.
   * Automated data transformation using Azure Synapse and Data Factory, enhancing efficiency and streamlining data workflows.
   * Spearheaded Master Data Management demo, focusing on Product, Customer and Supplier data, refining data governance & quality. 
   



