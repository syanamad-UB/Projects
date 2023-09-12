# Projects
This is **Susmitha Yanamadala**. I am currently pursuing **Masters in Data Science**. This repository consists of projects implemented as part of the course work and self-study.

**Project List and Details:**

**SQL:**

1. **IMDB Data Management and Analysis**:
   * Established a SQLite database, ensuring adherence to the third normal form for optimal table structuring.
   * Leveraged the "sqlite3" Python library for seamless table creation and data ingestion.
   * Conducted exploratory data analysis using SQL queries within Python, extracting key insights from the IMDB dataset.

2. **Bank Marketing Campaign Data Management and Web Integration**:
   * Initiated a PostgreSQL database, achieving table decomposition to BCNF for streamlined data handling.
   * Utilized the "psycopg2" Python library for robust table generation and effective data population.
   * Engaged in rigorous database testing and data analysis, harnessing commands like insert, update, delete, and SQL constructs such as joins, subqueries, CTEs, and triggers.
   * Executed in-depth query performance analysis; enhanced query speeds by optimizing via index creation.
   * Spearheaded the development of a real-time website interface, allowing direct data extraction from the database.

**Supervised & Unsupervised Machine Learning:**

1. **Heart Disease Prediction** (python):
   * Utilized vital signs like blood pressure, heart rate, cholesterol, and blood sugar, complemented by demographics and lifestyle factors, to forecast potential heart disease diagnoses.
   * Executed in-depth exploratory data analysis to gain nuanced understanding of the dataset.
   * Implemented a spectrum of classification models: Logistic Regression, Decision Tree, Random Forest, K Nearest Neighbor, Gaussian Naive Bayes, and Support Vector Classifier.
     
2. **Regression Analysis on Penguin & Flight Price Datasets** (python):
   * Spearheaded a machine learning project, crafting Logistic, Linear, Ridge Regression, and Ridge Regression with Gradient Descent models from scratch for the Penguins and Flight Price Prediction datasets, bypassing traditional frameworks like Scikit-learn.
   * Executed rigorous data visualization and cleaning on both datasets, unveiling pivotal insights and ensuring data integrity.
   * Deployed Logistic Regression on the Penguins dataset to forecast species, emphasizing hyperparameter tuning to refine model accuracy.
   * Leveraged the Flight Price Prediction dataset with Linear Regression, Ridge Regression, and Ridge Regression using Gradient Descent, delving into ticket price predictions while addressing model sensitivities and harnessing L2 regularization for enhanced generalization and robustness.
     
3. **Movie Clustering for Recommender Systems** (R):
   * Undertook data cleaning and pre-processing with a focus on features like popularity, vote metrics, and release timelines.
   * Deployed clustering algorithms, specifically K-means and hierarchical clustering, to group movies.
   * Envisioned application: Enhancing the precision and relevance of Movie Recommendation Systems through derived clusters.
  
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

1. **Multi Agent Reinforcement Learning** - Implemented a decentralized Multi-Agent Environment where agents work to rescue a person from fire. Employed tabular methods (Q-learning, SARSA) and deep RL methods (DQN, DDQN) in a custom grid world environment, analyzing and comparing their performance. Extended the evaluation to an existing collaborative MARL -  OpenAI particle environment (Simple Reference from Petting Zoo MPE) using DQN and DDQN,to assess their effectiveness in the context of the established environment.

