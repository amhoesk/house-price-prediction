# **Problem**

# Problem Description

A home is often the largest and most expensive purchase a person makes in their lifetime. Ensuring homeowners have a trusted way to monitor this asset is incredibly important. The goal is to develop a machine learning algorithm to give consumers as much information as possible about homes and the housing market, marking the first time consumers had access to this type of home value information at no cost.

# **Literature review**

More than 23 public repositories were found on GitHub on the topic of house price prediction and more than 252 competitions on kaggle. From dozens of academic publications, 15 articles which were published in high impact journals and were relevant to the problem were summarized as follows.

## Datasets Used for House Price Prediction

Studies have utilized various datasets to predict house prices, with the most commonly cited being the Boston Housing dataset, which contains 506 entries and 14 features. Other datasets mentioned include one with 1,460 records and 81 features, a dataset of real estate transactions in Taipei City and New Taipei City from 2017 to 2018, encompassing 92,857 transaction records after filtration, and a dataset of real estate transactions from nine French cities, spanning from 2015 to the first three quarters of 2019\. Additionally, researchers have collected data from realtor websites such as 99acres.com, magicbricks.com, and nobroker.com for Pune, India. Studies have also focused on specific cities or countries, including Singapore, Melbourne, Australia, Fairfax County, Virginia, Shanghai and Beijing, China, and London. A dataset of house listings in five Canadian cities \- Ottawa, Toronto, Mississauga, Brampton, and Hamilton \- was compiled using a web scraper, resulting in 10,418 listings with 57 features and the house advertising price. These diverse datasets demonstrate the wide range of data sources and geographic locations used in house price prediction research.

## Data Cleaning and Missing Data Handling

Studies described various techniques employed for data cleaning and handling missing values. A study focusing on predicting house prices in France emphasized the importance of data preparation, which includes removing inconsistencies, outliers, and missing values, as well as standardizing and encoding variables. They removed outliers based on the interquartile range, specifically eliminating transactions with prices above the third quartile plus one and a half times the interquartile range.  Missing values for land area, which were typically absent for apartment transactions, were replaced with zero. Another project utilizing data from a Canadian real estate listing website employed a comprehensive data cleaning process involving removing irrelevant features with over 50% missing values, merging related features, and simplifying categorical features with a high number of unique labels.  Missing values in the remaining features were handled on a case-by-case basis, with some replaced by zero, others imputed with the average value, and some filled using information from textual descriptions.

## Outlier Detection and Removal

Outliers can significantly impact the accuracy of house price prediction models, therefore, identifying and removing them is essential. A study focused on house price prediction using textual descriptions removed outliers representing locker or parking spaces based on their unusually low prices.  The researchers also identified and corrected other outliers caused by input errors through appropriate imputation methods.  Another study examining various machine learning algorithms for predicting house prices in France removed outlier price values using the interquartile range method.  The removal of outliers helps to ensure that the models are trained on data representing typical real estate transactions and are not skewed by extreme or erroneous values, leading to more accurate predictions.  This process is crucial for achieving reliable and robust house price prediction models.

## Important Features for House Price Prediction

Several studies have identified a range of influential features for predicting house prices, highlighting both property-specific and location-based attributes.  Structural features, such as living area, number of bedrooms and bathrooms, garage size, lot size, and property age, are consistently recognized as significant predictors. Location plays a pivotal role, with factors like proximity to amenities (schools, parks, hospitals, shopping centers), transportation hubs (airports, train stations), and major roads strongly influencing house prices. Additionally, neighborhood characteristics like socioeconomic indicators (crime rates, income levels) and environmental factors (noise levels, views) contribute to price variations.  Some studies have also explored the use of textual descriptions from real estate listings as features, extracting valuable information about property attributes and amenities.

### *Feature Engineering Techniques*

To enhance the predictive power of models, researchers have employed various feature engineering techniques.  Creating new features by combining existing ones has proven effective, such as calculating "TotalAreaSquareFeet" by summing the areas of different floors. Transforming numerical variables that are categorical in nature, such as MSSubClass (building class) and OverallCond (overall condition), aids in capturing non-linear relationships. Label encoding categorical variables with inherent order, like heating quality or kitchen quality, preserves ordinal information. One-hot encoding is widely used to transform categorical variables with multiple categories into binary features, enabling machine learning algorithms to process them effectively. Scaling numerical features using standardization or normalization techniques ensures that all features contribute equally to the model and prevents features with larger scales from dominating the learning process. Feature selection techniques like forward selection and backward elimination help identify the most relevant features, reducing model complexity and improving interpretability.

## Distribution of Key Features in House Price Prediction

Studies offer insights into the distribution of several significant features employed in house price prediction models. One study analyzing real estate transactions in Taipei City and New Taipei City observed that the oldest houses tend to be concentrated in the city center, while newer houses are more sparsely distributed in suburban areas.  The study also revealed that the most expensive houses are clustered near the city center, with cheaper houses spreading towards the suburban periphery. This pattern suggests a strong correlation between location, age, and price.  Another study utilizing data from a Canadian real estate listing website examined the distribution of the "bathroomTotal" feature, finding a higher concentration of one bathroom in lower-priced houses and a growing prevalence of three or four bathrooms in more expensive homes.  Similar trends were observed for features like "totalParkingSpaces" and "bedroomAboveGrade", indicating a relationship between these attributes and house price.  However, studies do not provide detailed analyses of the distributions of other features like the number of bedrooms or specific room sizes.

## A Comparative Analysis of Machine Learning Algorithms for House Price Prediction

Studies described a wide array of machine learning methods and algorithms employed for predicting house prices, encompassing both traditional and advanced techniques. Traditional approaches like Linear Regression, Decision Tree, and Support Vector Machine (SVM) have been widely used due to their simplicity and interpretability. Linear Regression, while straightforward, struggles with non-linear relationships and can be sensitive to outliers. Decision Trees, though easy to understand, are prone to overfitting and may not generalize well to unseen data.  SVMs are robust to outliers and effective in high-dimensional spaces, but they can be computationally expensive and sensitive to the choice of kernel function.

More advanced techniques, including Random Forest, Gradient Boosting, and Neural Networks have demonstrated superior performance.  Ensemble methods, which combine multiple models, have also gained popularity, including Stacked Generalization, Hybrid Regression, AdaBoost, XGBoost, LightGBM, and CatBoost. Random Forest stands out for its ability to handle high-dimensional data and robustness to overfitting. Gradient Boosting methods, especially XGBoost, are lauded for their accuracy, speed, and effective handling of missing values. Neural Networks, particularly Deep Learning models, excel at capturing complex non-linear relationships, but they require substantial computational resources and can be challenging to interpret. Studies consistently emphasize that the choice of the optimal algorithm is contingent upon factors such as dataset characteristics, feature engineering, and evaluation metrics.

## Evaluation Metrics for House Price Prediction Models

Numerous evaluation metrics have been employed to assess the performance of machine learning models for house price prediction.  Commonly used metrics include Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), R-squared (R²), Mean Squared Logarithmic Error (MSLE), and various quantiles of the prediction error distribution, such as Q1 (the first quartile). RMSE and MAE provide measures of the average prediction error in the original units of the target variable, with RMSE giving more weight to larger errors.  R² represents the proportion of variance in the target variable explained by the model, providing a measure of goodness of fit.  MSLE is particularly useful when dealing with large variations in target values, as it penalizes proportionally larger errors more heavily. Quantiles of the prediction error distribution offer insights into the spread and tail behavior of prediction errors, enabling a more comprehensive understanding of model performance.

The choice of the most appropriate evaluation metric depends on the specific goals of the study and the characteristics of the dataset. While RMSE is widely used and provides a measure of overall accuracy, it can be sensitive to outliers.  MAE is less susceptible to outliers but may not adequately capture the magnitude of large errors. R² offers a standardized measure of model fit, but it can be misleadingly high for models with a limited range of predictions. MSLE is suitable for datasets with large variances, as it is less sensitive to outliers in the high-value range. Ultimately, a combination of metrics is often recommended to provide a comprehensive evaluation of model performance and facilitate comparisons across different algorithms. Studies frequently highlight that neural network and ensemble learning methods, particularly Random Forest, XGBoost, and Gradient Boosting, consistently achieve superior performance across various evaluation metrics.  The best-performing model is often determined by the specific metric prioritized for the given task.

## References

1. Yalgudkar, Sushant Suresh, and N. V. Dharwadkar. "A Literature Survey on Housing Price Prediction." Journal of Computer Science & Computational Mathematics 12.3 (2022): 41-45.  
2. Adetunji, Abigail Bola, et al. "House price prediction using random forest machine learning technique." Procedia Computer Science 199 (2022): 806-813.  
3. Zulkifley, Nor Hamizah, et al. "House price prediction using a machine learning model: a survey of literature." International Journal of Modern Education and Computer Science 12.6 (2020): 46-54.  
4. Truong, Quang, et al. "Housing price prediction via improved machine learning techniques." Procedia Computer Science 174 (2020): 433-442.  
5. Basysyar, Fadhil M., and Gifthera Dwilestari. "House price prediction using exploratory data analysis and machine learning with feature selection." Acadlore Trans. AI Mach. Learn 1.1 (2022): 11-21.  
6. Fourkiotis, Konstantinos Panagiotis, and Athanasios Tsadiras. "Comparing Machine Learning Techniques for House Price Prediction." IFIP International Conference on Artificial Intelligence Applications and Innovations. Cham: Springer Nature Switzerland, 2023\.  
7. Ja’afar, Nur Shahirah, Junainah Mohamad, and Suriatini Ismail. "Machine learning for property price prediction and price valuation: a systematic literature review." Planning Malaysia 19 (2021).  
8. Wang, Pei-Ying, et al. "Deep learning model for house price prediction using heterogeneous data analysis along with joint self-attention mechanism." IEEE access 9 (2021): 55244-55259.  
9. Mohd, Thuraiya, et al. "An overview of real estate modelling techniques for house price prediction." Charting a Sustainable Future of ASEAN in Business and Social Sciences: Proceedings of the 3ʳᵈ International Conference on the Future of ASEAN (ICoFA) 2019—Volume 1\. Springer Singapore, 2020\.  
10. Geerts, Margot, and Jochen De Weerdt. "A survey of methods and input data types for house price prediction." ISPRS International Journal of Geo-Information 12.5 (2023): 200\.  
11. Zhang, Hanxiang, Yansong Li, and Paula Branco. "Describe the house and I will tell you the price: House price prediction with textual description data." Natural Language Engineering 30.4 (2024): 661-695.  
12. Tchuente, Dieudonné, and Serge Nyawa. "Real estate price estimation in French cities using geocoding and machine learning." Annals of operations research 308.1 (2022): 571-608.  
13. Sharma, Hemlata, Hitesh Harsora, and Bayode Ogunleye. "An Optimal House Price Prediction Algorithm: XGBoost." Analytics 3.1 (2024): 30-45.

# **Proposed Solution**

Some variables, such as listing price, `lp_price`, are highly correlated with the target feature. The tax, `taxes`, feature can also be assumed to be unavailable since it can be calculated from the house price given the location of the house such as municipality and province. However, it is assumed that these variables, along with features like provincial assessment house prices, are not typically available. While the problem could be reframed to analyze how much a house's price deviates from its assessment or listing price, that would represent a distinctly different challenge. 

As outlined in the problem definition, the emphasis of this task is on the data pipeline and thinking process. Therefore, the objective of this solution was assumed to predict the target `price_sold` feature of a house based on its intrinsic features, such as the number of bedrooms, bathrooms, and similar attributes, rather than relying on variables that are either correlated with or derived from the house price.

# Data Processing

The following table explains each feature with its corresponding descriptive statistics. The raw data includes 100,000 observations with 95.6% of rows (95,592) having missing values emphasizing on the importance of a crucial data cleaning step in building a good machine learning model. 

| Feature Name | Explanation | count | missing | unique | top | freq | min | mean | median | max | std |
| ----- | ----- | :---: | :---: | :---: | :---: | :---: | :---: | ----- | ----- | ----- | ----- |
| **property\_type** | The type of property (e.g., house, apartment, condo). | 100000 | 0 | 32.0 | D. | 47047 |  |  |  |  |  |
| **br** | The number of bedrooms in the property. | 98830 | 1170 |  |  |  | 0.0 | 2.77 | 3 | 11 | 1.16 |
| **br\_plus** | Additional rooms that could serve as bedrooms (e.g., den or flex spaces). | 40866 | 59134 |  |  |  | 0.0 | 1.21 | 1 | 9 | 0.57 |
| **br\_final** | Total number of bedrooms, including additional rooms. | 100000 | 0 |  |  |  | 0.0 | 2.83 | 3 | 11.4 | 1.15 |
| **bath\_tot** | The total number of bathrooms in the property. | 99997 | 3 |  |  |  | 0.0 | 2.53 | 2 | 18 | 1.30 |
| **taxes** | Annual property taxes for the property. | 73911 | 26089 |  |  |  | $0 | $4,031 | $3,474 | $694,557 | $7,636 |
| **lp\_dol** | Listing price of the property in dollars. | 100000 | 0 |  |  |  | $0 | $570,960 | $459,900 | $89,999,900 | $790,303 |
| **yr\_built** | The year the property was constructed. | 44785 | 55215 | 10.0 | 0-5 | 12307 |  |  |  |  |  |
| **gar\_type** | The type of garage (e.g., attached, detached, none). | 85315 | 14685 | 8.0 | Attached | 34394 |  |  |  |  |  |
| **garage** | The number of garage spaces available. | 99013 | 987 |  |  |  | 0.0 | 1.27 | 1 | 330 | 2.01 |
| **topHighschoolScore** | The score of the highest-rated high school near the property. | 99885 | 115 |  |  |  | 0.0 | 4.28 | 5 | 9.9 | 3.19 |
| **topBelowHighschoolScore** | The score of the highest-rated below high school near the property. | 99885 | 115 |  |  |  | 0.0 | 5.67 | 5.6 | 10 | 2.15 |
| **geo\_latitude** | The geographic latitude coordinate of the property. | 99999 | 1 |  |  |  | \-33.87 | 43.80 | 43.74 | 56.23 | 0.53 |
| **geo\_longitude** | The geographic longitude coordinate of the property. | 99999 | 1 |  |  |  | \-127.59 | \-79.49 | \-79.44 | 151.21 | 1.87 |
| **lot\_frontfeet** | The width of the lot's frontage in feet. | 58078 | 41922 |  |  |  | 0 | 69 | 42 | 103911 | 494 |
| **lot\_depthfeet** | The depth of the lot in feet. | 56147 | 43853 |  |  |  | 0 | 159 | 113 | 432213 | 1843 |
| **sqft\_numeric** | The total square footage of the property. | 66071 | 33929 |  |  |  | 499 | 1591 | 1300 | 9310 | 994 |
| **id\_community** | Identifier for the community or neighborhood where the property is located. | 99996 | 4 |  |  |  |  |  |  |  |  |
| **id\_municipality** | Identifier for the municipality where the property is located. | 99996 | 4 |  |  |  |  |  |  |  |  |
| **date\_start** | The start date of the property's listing. | 100000 | 0 |  |  |  | 2012-09-13 | 2016-07-18 |  | 2016-10-26 |  |
| **date\_end** | The end date of the property's listing. | 100000 | 0 |  |  |  | 2015-10-16 | 2016-08-18 |  | 2019-11-13 |  |
| **price\_sold** | The final selling price of the property in dollars. | 70731 | 29269 |  |  |  | $0 | $516,197 | $449,900 | $25,888,888 | $560,700 |

## ![][image1]

###### *Fig 1- The location of listings*

## Duplicate values

43 duplicate values were found among 100k observations.

## Missing values

Some important features have a significant amount of missing data.

+ `price_sold` (target variable) is an important feature but there are 29,269 missing values. Since the correlation between `price_sold` and `lp_dol` is 0.97 and there are no missing values for `lp_dol`, missing values of `price_sold` were replaced by `lp_dol` using a linear regression model.  
+ `sqft_numeric` is critical for prediction, and imputation may introduce significant noise. Moreover, there is no correlation between `sqft_numeric` and `lot_frontfeet*lot_depthfeet`. So for algorithms that do not support missing values, corresponding rows should be dropped.  
+ `bath_tot` is an important feature and cannot be ignored. Although the missing values could be estimated, considering that there are only three missing values, corresponding rows were removed to save time.  
+ While there are many missing values for `br` and `br_plus` feature, there are no missing values for `br_final` feature. Since `br_final` were calculated based on `br` and `br_plus`, the missing values can be reconstructed.

![][image2]

###### *Fig 2- Distribution of missing values over different features*

## Removing outliers

* Property type:  
  * Any property type with less than 50 occurrences were removed.  
  * For simplicity, property types with less than 1000 occurrences were merged under `other` property types.  
* The distribution of `price_sold` was strange with 8,000 values under 50,000 while there were no values between 50,000 to 200,000. Values under 50,000 were removed.  
* House Price: Rows with house price more than 10 million very removed.  
* Area: Rows with sqft more than 7k very removed.  
* Total Bathrooms: Rows with bathrooms more than 10 were removed.  
* Total Garage: Rows with more than 10 were removed.  
* Garage type has a value named `Underground` that should be renamed to `Undergrnd` value.

| Before removing outliers | After removing outliers |
| ----- | ----- |
| ![][image3] | ![][image4] |

###### *Fig 3- Boxplot of features before and after removing outliers*

## Feature normalization

Numerical features like `bath_tot`, `br_plus`, `topBelowHighschScore`, `topHighschScore`, `br`, `sqft_numeric`, `garage` were normalized between 0-1 using `MinMaxScaler`.

# Exploratory Data Analysis

The following figure shows the histogram of numerical features and their pairwise relationship. For example, there is a strong positive correlation between the `price_sold` and `bath_tot` and `sqft_numerc`.

![][image5]

###### *Fig 4- Pairwise relationship between features*

# Model Selection

Two well-known models from the literature namely Random Forest and XGBoost were chosen and trained.

# Model Training

In order to get the best of hyper-parameters for the model, a hyper-parameter tuning approach was used for each model. Categorical variables such as `property_type` were encoded to numerical values using scikit-learn `LabelEncoder` class.

# Model Evaluation

Four evaluation metrics were chosen based on the literature:

1. Root Mean Squared Error (RMSE)  
2. Mean Absolute Error (MAE)  
3. R-squared (R²)  
4. Mean Squared Logarithmic Error (MSLE)

# **Results and Discussion**

The root mean squared error (RMSE) metric shows a significantly higher error compared to the mean absolute error (MAE), suggesting that the model struggles to accurately predict high-priced houses. This observation aligns with findings reported in other studies. It highlights the need for specialized modeling approaches or expert case-by-case analysis when dealing with expensive properties. The R² metric indicates that approximately 81% of the variability in house prices is explained by the model's features, which is reasonable given the limited number of features and the quality of the dataset. Additionally, the mean squared logarithmic error (MSLE) metric achieves a value below 0.1 (\~0.04), typically considered indicative of good model performance.

Across all metrics, XGBoost outperformed Random Forest, demonstrating lower errors and a higher R² value, confirming its superior predictive performance.

| Model |  RMSE(×1000 $) | MAE(×1000 $)  | R²(%) | MSLE |
| ----- | :---: | :---: | :---: | :---: |
| **Linear Regression** | 389 | 230 | 60.6 | NA |
| **Random Forest** | 284 | 105 | 80.6 | 0.038 |
| **XGBoost** | 283 | 102 | 80.7 | 0.035 |

Performance metrics were found to be sensitive to listings prices showing low prediction power for expensive house prices. Limiting the range of listing prices to $200K-2M (\~3.9% of listings are between 2M-10M) improves the model's performance significantly.

| Model |  RMSE(×1000 $) | MAE(×1000 $)  | R²(%) | MSLE |
| ----- | :---: | :---: | :---: | :---: |
| **Linear Regression** | 210 | 152 | 64.7 | 0.107 |
| **Random Forest** | 127 | 74 | 87.8 | 0.029 |
| **XGBoost** | 121 | 71 | 88.8 | 0.027 |

## Limitations and Potential Improvements 

The performance of the model could be significantly enhanced with additional time to thoroughly review and compare relevant literature. The volume and quality of the dataset are critical factors influencing model performance. In particular, more comprehensive data preprocessing and feature engineering are necessary. The primary focus during this project was on removing anomalies and working with easily manageable features to expedite the completion of the data pipeline. However, the following considerations were overlooked, largely due to the limited timeframe:

* **Imbalanced Dataset:** The dataset was unbalanced regarding the number of observations across different property types, potentially leading to poor predictions for certain categories of houses. A more detailed analysis is required, including error plots against property types and other features, to identify sources of low-quality estimations. Addressing this issue may involve oversampling underrepresented property types or applying weighted samples to balance the dataset.  
* **Better outlier detection algorithm:** I addressed outliers by simply removing rows with extreme feature values. While straightforward, this approach was often naive and likely imprecise in many instances. A more sophisticated alternative would involve using clustering-based algorithms to identify outliers. However, such methods typically require additional parameters, which must be carefully tuned to ensure accuracy and effectiveness.  
* **Limited Algorithm Scope:** To obtain results quickly, only a few well-known machine learning algorithms were utilized. Exploring more advanced techniques, such as deep learning models, could likely yield better performance outcomes.

These improvements would provide a more robust and reliable model, addressing current limitations and enhancing predictive accuracy.