# Duplicate-Questions-Remover

This project is used to find out list of different unique questions from a large list of questions having many similar and duplicate questions. This problem is inspired from kaggle competition where we need to identify if pair of given Quora questions are similar/duplicate or not.

The project can be applied in various places and organization in several applications, for eg: 
                
                >reviewing and obtaining set of questions asked related to a product/service on e-commerce website
                >for finding out commonly asked questions on discussion forums and websites
                >for filtering out large number of queries
                >for filtering out similar reviews/comments on products and services
                
These all functionalities and applications can result in lesser human effort and more time savings by automating the process of literature review and segregation of large amount of textual data into smaller set of unique data where there is high probability of having large number of similar and duplicate entries.

The project uses concepts of Natural Language Processing (NLP), Machine Learning, Deep Learning, Data Analysis, Boosting and Ensemble Machines to predict the unique set of questions.

In order to achieve maximum accuracy after several testing on several models, it was taken into consideration that ideal model to predict the output of distinct question that a combined model of 5 different classifiers is to be used; with input of 28 different features obtained from a pair of questions. These 28 features are obtained by applying various string and string-vector manipulations and features; and various concepts of NLP like string comparison distances,fuzzy features etc. The five models used are:
                    
                    I. XGBosst Classifier
                    II. Light GBM Classifier
                    III. Artificial Neural Network (ANN) Classifier (with 5 hidden layers)
                    IV. Random Forest Classifier
                    V. Catboost Classifier
   
Since the classifiers are binary classifiers (0: different questions; 1: similar/duplicate questions); Mode of outputs of the above 5 classifiers are taken to calculate final output.

The input is a file containing list of all questions. Output file contains reduced distinct questions

There are 4 files in this project:
        
        I. data_preprocessing.py: To add 28 features used for predictions of a pair of different questions
        II. models.py: to create, train and store 5 above mentioned models
        III. data_framer.py: To create ideal data frame for preprocessing and prediction from input single columned datframe of all                                   questions; To create single-columned dataframe of output file
        IV questions_reducer.py: Combines functionality of all 3 above files with calcuating the final output of predictions; Storing                                     final result in output file: unique_questions.csv
        
        
Important Libraries to be pre-installed:
        
        1.Tensorflow
        2.Pandas
        3.Numpy
        4.NLTK
        5.Statics
        6.Genism
        7.Joblib
        8.Fuzzywuzzy
        9. Pyemd
        10.Scipy
        11. xgboost
        12.Sklearn
        13.lightgbm
        14.catboost
        15.python-Levenshtein
        
Along with thatwe need Word2vec mode.We download the GoogleNews-vectors-negative300.bin.gz binary and use Gensim’s load_Word2vec_format function to load it into memory. (1.5 GB compressed file: Actual size on loading (>3GB))

EXTRA NOTES:
    
    >Pre processed data for training is stored in DataSset/training_data  in model_inputs.rar
    >Original Data set download link is provided in DataSset/training_data
    >Pre-trained models are present in \models
    >Random forest model is compressed in rf_model.rar (For ease of uploading on Git)
    >\DataSet contains sample.csv(file of 99 question) and its output csv file(having 37 questions) for reference
