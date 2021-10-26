# COVID-19 CONFIRMED CASES FORECASTING IN USA USING TIME SERIES ANALYSIS
COVID-19 has created a havoc in the world with new cases everyday springing up everywhere. Being said that, there also has been a challenge for maintaining the data as to how many have been reported, how many deaths occurred etc.
The need for maintaining the data comes from the goal of a forecast as to how will the situation will look like in near future.

In this project, real time data of COVID-19 confirmed cases in the USA have been collected and fit in a time series model in order to forecast the future.
Data provided is from 22-Jan-2020 to 23-Sep-2021.

In the original raw file, cases as per state and other columns such as latitude, longitude, province etc. are provided.
For our ease, the number of cases is added from each state to show the total cases in the whole USA as required. Other redundant columns have been removed. These operations have been done in Excel only.

After necessary preprocessing, three time series models have been selected for making a forecast. At the end, whichever model shows the lowest rmse value and almost a best fit of the current trend or seasonality will be adjudged the best model.
