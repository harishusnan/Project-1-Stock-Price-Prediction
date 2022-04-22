# Project-1-Stock-Price-Predictions


##### Project Description

Develop a supervised machine learning model using Linear Regression (LR) method and Exponential Moving Average (EMA) as a technical indicator


|Item|Link|
|---|---|
|Data Source (kaggle)|[act_bliz](https://www.kaggle.com/datasets/psycon/game-companies-historical-stock-price-2022-04?select=act_bliz.csv)|
|Project Source Code|[Stock Prices Prediction LR Model](https://github.com/harishusnan/Project-1-Stock-Price-Prediction/blob/main/Blizzard%20LR%20Model.ipynb)|

##### Problem Statement

Mr Z is a huge fan of Blizzard company and he has an extra money for an investment. He want to invest in a company with **high reputation** and the **price movement** of its stock prices is **strong uptrend**. It is important for him to analyze the company performance and its stock price pattern before he started to invest his money. Furthermore, identifying the **right time to buy the stock** is his main goal once recognizing the company's stock price pattern.



*Let us help him out!*

We assume that Mr Z has already satisfied with the company overall performance or fundamental analysis (FA). So, we will proceed with analyzing the price pattern and developing a future price prediction model to help Mr Z determine the right time to buy the stock. In this case, we want to apply linear regression (LR) technique to develop our machine learning model. We focus on developing the LR model based on the historical data provided and we also included technical indicator in training our model to predict future prices.



*Let's get started!*


##### Import Packages & Read Data

First thing first, we need to import python packages and load our data before we start analyzing and developing our model.

```python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = %pwd

histdata = pd.read_csv(path + "dataset/act_bliz.csv")

histdata.head()

```

![Blizzard Dataset](https://github.com/harishusnan/Project-1-Stock-Price-Prediction/blob/main/images/Dataset.png)

##### Data Cleaning & Preparation

Looks like we have successfully uploaded our dataset. Now, we apply exploratory descriptive analysis (EDA) method to analyze the data through ```info()``` and ```describe()``` method. 

![Dataset describe](https://github.com/harishusnan/Project-1-Stock-Price-Prediction/blob/main/images/Dataset_describe.png)
![Dataset info](https://github.com/harishusnan/Project-1-Stock-Price-Prediction/blob/main/images/Dataset_info.png)

Great! There are no missing values in our dataset. Good quality of dataset will helps us in generating a model with high accuracy.

Next, let us create a new dataset that contains only *Date (Index)* and *Close* columns. We need to identify if there are outliers in our dataset and we also want to see the price movement pattern.

![Boxplot](https://github.com/harishusnan/Project-1-Stock-Price-Prediction/blob/main/images/Boxplot.png)
![Price_movement](https://github.com/harishusnan/Project-1-Stock-Price-Prediction/blob/main/images/Price_Movement.png)

We can conclude that there are **no outliers exist** in our dataset and the price movement is **strong uptrend**.
*So far, so good!*

###### Technical Indicator

As mentioned above, we will train our model using a technical indicator. Technical indicator is commonly used by experts for future price predictions. There are a lot of technical indicators available such as SMA,EMA,MACD and so on. In our case, we'll use Exponential Moving Average (EMA) as our technical indicator. 

Fortunately, python has a module called pandas_ta which consists of list of technical indicators for technical analysis (TA) purposes. To apply EMA in our analysis, we need to specify the number of period that we want to calculate its average values. So, let's decided to calculate the EMA over 10 days period and add the values into new column in our dataset.

```

import pandas_ta

df.ta.ema(close="Close",length=10,append=True)

df.head(10)

```

![EMA Dataset](https://github.com/harishusnan/Project-1-Stock-Price-Prediction/blob/main/images/Dataset_EMA10.png)

Since we have NaN values in our dataset, we need to clean it first before we proceed to the next steps. So, let's just drop the NaN values as it will not give a huge impact to our dataset as our dataset is quite large. NaN values existed due to no preceeding values from which EMA function could be calculated. EMA function calculated the average value by getting the values from Day 1 - Day 10 divided by count = 10 (days) and display on 10th row.

![EMA Dataset Cleaned](https://github.com/harishusnan/Project-1-Stock-Price-Prediction/blob/main/images/Dataset_EMA10_cleaned.png)
![Actual vs EMA10 Line Chart](https://github.com/harishusnan/Project-1-Stock-Price-Prediction/blob/main/images/Actual_vs_EMA10_Linechart.png)

EMA-10 predicts quite well the price of the stock. The plot shows that there is only a small difference between actual and EMA-10 values. Thus, we can proceed with the model development.



##### Model Development & Evaluation

###### Split Dataset

Common practice of developing machine learning models is to split dataset into training and testing data. We need to ensure that training dataset has higher amount of data than testing dataset. In this case, we apply 80/20 partition to form our training and testing data set. 80% of our data will be used for training and we will test our model using the remaining 20% of our data.

```

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(df[["Close"]],df[["EMA_10"]],test_size=0.2)

```

###### Build Model

Building models in python is quite simple as we can utilize the python packages provided by the developer. For linear regression model, we use ``` LinearRegression() ``` to build the model.

```

from sklearn.linear_model import LinearRegression

#set the model
model = LinearRegression()

#train the model
model.fit(x_train,y_train)

#test model with 20% samples
y_pred = model.predict(x_test)

```


Done! Now, we have trained our model and also generated predicted values. Next, we should evaluate our model performance to see how well the model fits our data. We will use mean absolute error (MAE) and coefficient of determination (r2).


###### Model Evaluation

```

#evaluation metrics
#lower MAE is better
#closer model correlation coefficient to 1.0 the better
print("Model coefficient: ", model.coef_)
print("Model mean absolute error: ", mean_absolute_error(y_test,y_pred))
print("Model R squared error: ", r2_score(y_test,y_pred))

```

![Evaluation Metrics](https://github.com/harishusnan/Project-1-Stock-Price-Prediction/blob/main/images/Evaluation_metrics.png)

We can conclude that our linear regression model fits our data very well. Awesome!




##### Data Visualization and Interpretation

Let's plot our Predicted values against Adjusted Close (EMA-10) values and observe the comparison.

```

#convert array to dataframe
y_pred = pd.DataFrame(y_pred)

#adding column name
y_pred.columns = ["Predicted"]

#combine dataframe y_pred to y_test
y_test['Predicted'] = y_pred['Predicted'].values

sns.lmplot(x='EMA_10',y='Predicted',data=y_test,order=2, line_kws={'color':'red'} ,height=10,aspect=1.2)
plt.xlabel("Actual EMA-10",fontsize=15)
plt.ylabel("Predicted",fontsize=15)
plt.show()

```

![Predicted vs EMA10 Scatter Plot](https://github.com/harishusnan/Project-1-Stock-Price-Prediction/blob/main/images/Predicted_vs_EMA10.png)



Let's see in bar plot as well:

![Predicted vs EMA10 Bar Plot](https://github.com/harishusnan/Project-1-Stock-Price-Prediction/blob/main/images/Predicted_vs_EMA10_barchart.png)


We can clearly see that it looks like a pretty good fit. 

So far, we have developed based on the historical data provided (we took Close and EMA-10 values) and evaluated our model performance. Our model accurately predict the closing price on that day based on the EMA of any given day (refer to the bar chart below).

Let's add a new column which consists of labels of *Buy* or *Sell* or *Hold* to help Mr Z to identify when should he buy/sell/hold his shares. We subtract Open Price column with Predicted Price column and labeled the output. The conditions are:

- When Open Price < Predicted Price == *Buy*
- When Open Price > Predicted Price == *Sell*
- When Open Price = Predicted Price == *Hold*

![Final Table](https://github.com/harishusnan/Project-1-Stock-Price-Prediction/blob/main/images/Final_table.png)



That's it! We have completely developed our linear regression model for stock prices prediction. We can try to develop stock prices prediction model using other machine learning algorithm in the future.

Thank you for reading my data science project. See you in the next project!




