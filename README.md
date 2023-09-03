## Table of Contents: 
- [Problem Statement](https://github.com/arezazadeh/CapstoneProjectOnDiabetes#Problem-Statement)
- [Features Being Used](https://github.com/arezazadeh/CapstoneProjectOnDiabetes#Features-Being-Used)
- [1.1 EDA](https://github.com/arezazadeh/CapstoneProjectOnDiabetes#EDA)
    - [1.1.0 Correlation Matrix](https://github.com/arezazadeh/CapstoneProjectOnDiabetes#110-correlation-matrix)
    - [1.1.1 Age Analysis](https://github.com/arezazadeh/CapstoneProjectOnDiabetes#Age-Univariate-and-BivariateAnalysis)






## Problem Statement

1. **You are diabetic**: This is a direct classification result when the probability of the diabetic class is above a certain threshold, often 0.5 for binary classification. However, this threshold can be adjusted based on the specific needs of the model and the business or medical requirements.

2. **You are not Diabetic**: Similarly, this is another direct classification result when the probability of the non-diabetic class is above the threshold.

3. **You have x% risk to become diabetic**: This is essentially the probability value assigned by the model to the diabetic class for the given entry. For instance, if you input a patient's data into the model and it outputs a probability of 0.65 for the "diabetic" class, it can be interpreted as "This patient has a 65% risk of becoming diabetic."

**Using Selected/Best Model with Probabilities:**

Using the model to predict class probabilities using the `predict_proba` method. This will provide probabilities for each class, and in the context of a binary problem like yours, the output will have two columns: the first column represents the probability of the "non-diabetic" class, and the second represents the probability of the "diabetic" class.

**to utilize models to analyze diabetes and non diabetes patients to find out what are the leading causes for diabetes**



## Features Being Used

Certainly! Here's a brief description of each feature:

1. **gender**: 
    - Represents the biological sex of the individual.
    - Typically categorized as "male," "female," or in some datasets, "other" or "non-binary."

2. **age**: 
    - Represents the age of the individual, usually measured in years.
    - Can be a continuous variable (e.g., 25.5 years) or discrete (e.g., 25 years).

3. **hypertension**: 
    - Indicates whether the individual has hypertension (high blood pressure).
    - It's usually a binary indicator: "yes" (or 1) for those with hypertension, and "no" (or 0) for those without.

4. **heart_disease**: 
    - Indicates whether the individual has any heart-related diseases.
    - Typically a binary feature: "yes" (or 1) for those with heart disease and "no" (or 0) for those without.

5. **smoking_history**: 
    - Describes the individual's history with smoking.
    - Can be categorical, with values such as "never smoked," "formerly smoked," "currently smoking," etc.

6. **bmi**: 
    - Stands for "Body Mass Index."
    - A measure used to determine whether a person has a healthy body weight for their height.
    - Calculated as weight in kilograms divided by the square of height in meters. 
    - Used as an indicator of underweight, normal weight, overweight, or obesity.

7. **HbA1c_level**: 
    - Refers to the Hemoglobin A1c test.
    - Measures the average blood sugar (glucose) level over the past 2-3 months.
    - Used to diagnose diabetes and gauge how well someone is managing their diabetes.
    - Values are often presented as a percentage. A higher percentage can indicate higher blood glucose levels.

8. **blood_glucose_level**: 
    - Represents the concentration of glucose present in the blood.
    - Measured in milligrams per deciliter (mg/dL) or millimoles per liter (mmol/L).
    - A key indicator for diagnosing and monitoring diabetes.

Each of these features provides valuable health-related information. In the context of predicting or understanding diabetes, they are all relevant as they are associated with risk factors or direct indicators of diabetes.



## 1.1.0 Correlation Matrix 

<img src="images/corr.png">


Here's what we can infer from the correlation matrix:

### Age
- **Moderate positive correlation with BMI (0.34)**: As age increases, BMI also tends to increase.
- **Low positive correlations with hypertension (0.26), heart disease (0.24), and diabetes (0.26)**: Older people are more likely to have these conditions, but the correlation isn't very strong.
  
### Hypertension
- **Low positive correlations with age (0.26), diabetes (0.20), and BMI (0.15)**: People with hypertension are slightly more likely to be older, diabetic, and have a higher BMI.
  
### Heart Disease
- **Low positive correlations with age (0.24) and diabetes (0.17)**: Older individuals and those with diabetes are slightly more likely to have heart disease.
  
### BMI
- **Moderate positive correlation with age (0.34) and low positive correlation with diabetes (0.21)**: Higher BMI is more common in older individuals and those with diabetes.
  
### HbA1c Level
- **Moderate positive correlation with diabetes (0.41)**: Higher levels of HbA1c are strongly associated with diabetes.
- **Low positive correlation with blood glucose level (0.17)**: Higher levels of HbA1c are somewhat associated with higher blood glucose levels.
  
### Blood Glucose Level
- **Moderate positive correlation with diabetes (0.42)**: Higher blood glucose levels are strongly associated with diabetes.
- **Low positive correlation with HbA1c level (0.17)**: Higher blood glucose levels are somewhat associated with higher HbA1c levels.
  
### Diabetes
- **Moderate positive correlation with HbA1c level (0.41) and blood glucose level (0.42)**: People with diabetes are likely to have higher levels of HbA1c and blood glucose.

To summarize, the strongest correlations we have are between diabetes and HbA1c levels, and between diabetes and blood glucose levels. These could be key features if we are looking to predict or understand diabetes in this dataset. Age and BMI also show moderate positive correlations with diabetes.


