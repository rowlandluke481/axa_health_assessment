# Customer Support Sentiment and Call Outcome Analysis
This project uses the GPT-4-0 mini model to analyse customer sentiment and call outcomes based on customer support transcripts. The goal is to provide actionable insights into customer satisfaction and support performance, with a focus on understanding the sentiment and outcome classifications in customer interactions.

## Project Overview
The project leverages a GenAI model to perform sentiment and outcome classification on customer support conversation transcripts. Key areas of analysis include:

* Sentiment Analysis: Identifying customer sentiment in support conversations.
* Outcome Prediction: Classifying call outcomes based on conversational cues.
  
These insights aim to enhance customer support performance and inform support process improvements.

## Key Features
### Model Performance
* Sentiment Classification: Achieved perfect classification performance (F1 score: 1.00).
* Outcome Prediction: High accuracy in outcome classification, with an F1 score of 0.85.
### Analysis Capabilities
* Sentiment Distribution: Visualised distribution across positive, negative, and neutral classes.
* Outcome Patterns: Uncovered call resolution and follow-up trends based on conversation features.
* Feature Importance: Identified significant variables in customer support outcomes, such as sentiment tone and interaction frequency.

1. Real-time sentiment monitoring for immediate issue escalation.
2. Detailed tracking of customer issues and agent responses.
3. Proactive support strategies based on early sentiment cues.
## Future Steps
* Real-Time Analysis: Implement real-time sentiment and outcome monitoring.
* Entity Extraction: Extend extraction to track call duration, member and claim numbers, and repeated interactions.
* Dynamic Sentiment Tracking: Explore multi-turn conversation analysis to capture sentiment shifts.
* Enhanced Fairness Metrics: Evaluate model fairness across demographic groups.
* Gold Standard Test Data: Create human-labeled data for continuous model validation.

# Mental Health Prediction Project
This repository contains the code and analysis for predicting mental health outcomes using the Depression Dataset from Kaggle. The project was developed as part of a data science exercise to evaluate exploratory data analysis (EDA), predictive modeling, and approaches for handling model bias and class imbalance.

## Project Overview
### Objective
The primary goal of this project is to:

1. Extract valuable insights from the dataset using exploratory data analysis.
2. Build predictive models to identify individuals at risk of mental illness, with a focus on the target variable, "History of Mental Illness."
3. Address model performance, bias, and limitations, proposing improvements for future work.

## Dataset
The Depression Dataset is required to run the analysis. Due to data privacy, it is not stored in this repository. Please download the dataset from Kaggle and place it in the data/ directory.

## Exploratory Data Analysis (EDA)
The initial analysis focuses on understanding correlations between variables such as socioeconomic status, family history, and lifestyle factors in relation to mental health outcomes.

Key findings:

* Significant Predictors: Family history, income level, and lifestyle habits (e.g., physical activity, smoking) showed notable correlations with mental health risk.
* Data Imbalance: There was a significant class imbalance, which necessitated the use of SMOTE to improve model sensitivity for at-risk cases.

## Modeling Approach
The model selection process considered multiple algorithms, including logistic regression, random forest, and XGBoost. The performance was evaluated using metrics like accuracy, F1 score, and AUC.

## Model Performance
Results on the test set after tuning:

* Accuracy: 63%
* F1 Score: 0.28 for the illness class, showing room for improvement
* AUC Score: Averaged around 0.59 across cross-validation
  
## Addressing Model Bias and Class Imbalance
To handle bias and improve fairness:

* SMOTE was applied to address class imbalance.
* Socioeconomic variables were reviewed for potential bias, and exclusion of certain features was considered to reduce risk of perpetuating stereotypes.
## Limitations and Next Steps
### Limitations
* Class Imbalance: The model’s recall for illness prediction remained low despite SMOTE.
* Data Quality: Certain predictors have limited variability, which may limit generalizability.
### Future Improvements
* Experiment with alternative algorithms (e.g., neural networks, ensemble methods) for higher sensitivity.
* Collect more balanced and representative data to reduce bias in prediction outcomes.
