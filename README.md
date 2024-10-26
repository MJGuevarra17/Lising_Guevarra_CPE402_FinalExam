# Analysis of Glacier Mass Balance Over Time

This project aims to analyze the changes in glacier mass balance over time, exploring the trends and patterns in the data to understand the impact of climate change on glaciers. By employing various regression models, we seek to forecast future mass balance changes based on historical data.

## Dataset Description

The dataset used in this project contains information on glacier mass balance measurements from various locations worldwide. It includes columns such as "Year" and "Mean cumulative mass balance," which represent the time frame of the data and the average mass balance of glaciers, respectively. This dataset is sourced from the Glacier Mass Balance Database.

## Summary of Findings

The analysis reveals a significant decline in glacier mass balance over the years, indicating a trend of melting and reduced ice accumulation. The models evaluated demonstrate varying degrees of accuracy in predicting mass balance changes, with linear regression showing a strong correlation with the observed data.

## Data Preprocessing

Data preprocessing involved checking for missing values and removing any incomplete records to ensure data integrity. The dataset was then examined for data types, and a statistical summary was provided to understand the distribution of values across different features.

## Exploratory Data Analysis

### Visualization

1. Glacier Mass Balance Over Time

![alt text](https://imgur.com/jNaAeCa)

This line plot illustrates the trend of glacier mass balance over the years, highlighting a general decline in mass balance.
Year-over-Year Change in Glacier Mass Balance.

2. Year-over-Year Change in Glacier Mass Balance

![alt text](https://imgur.com/ndkmbFU)

The bar plot reveals fluctuations in the annual mass balance of glaciers over time, showcasing both positive and negative changes. In recent years, there is a notable trend of declining mass balance, with several years exhibiting significantly negative changes, indicating accelerated glacier loss. These patterns suggest that climate factors are increasingly impacting glacier stability, leading to more pronounced annual losses. The variability in mass balance changes also reflects the influence of different climatic conditions, such as variations in temperature and precipitation. Overall, the plot underscores the urgency of addressing climate change, as the continued decline in glacier mass balance poses serious implications for ecosystems and water resources.

3. Correlation Heatmap

![alt text](https://imgur.com/5SdNZst)

This heatmap reveals the relationships between different features in the dataset, showing strong correlations among certain variables that can influence mass balance. There is a strong negative correlation of -0.96 between the year and mean cumulative mass balance, indicating that as time progresses, glacier mass balance decreases. A positive correlation of 0.92 between the year and number of observations suggests that more recent years have more recorded observations. The mean cumulative mass balance positively correlates at 0.75 with annual change in mass balance, showing that years with higher balances also have greater annual changes. Conversely, a negative correlation of -0.78 between mean cumulative mass balance and the number of observations implies that higher balances correspond to fewer observations. Lastly, the annual change in mass balance negatively correlates at -0.68 with the year, indicating a trend of increasing glacier loss over time.

## Model Development

The model development process involved selecting multiple regression algorithms, including Linear Regression, Decision Tree, Random Forest, and Support Vector Machine. The dataset was split into training and testing sets to train the models and evaluate their performance effectively.

## Model Evaluation

The model evaluation was conducted using metrics such as Mean Squared Error (MSE) and R-squared (R²). These metrics provide insights into the accuracy and reliability of the models in predicting glacier mass balance changes. Each model was assessed based on its performance, with Linear Regression achieving the best results in this analysis.

## Conclusion

The project successfully highlighted the decline in glacier mass balance over the years and provided insights into the effectiveness of various regression models in predicting this trend. The findings underscore the need for ongoing monitoring and research into the impacts of climate change on glacial regions, as well as the importance of utilizing historical data to inform future climate models.

## Contributors

❗ NOTE: Your professor be the one to fill this section.
