# The Right Price - A London Airbnb Recommender
###### Galvanize Final Capstone Project &bull; Lauren Schoener &bull; 01/2020

&nbsp;
## Motivation

Like many other people in their mid to late twenties, I became interested in taking the plunge on a first home. Housing prices can be steep, and if you do not plan to live somewhere all year round or if you have extra rooms in your home you are not using, it is very tempting to rent that extra space out on a platform like Airbnb. But how much money can you expect to make renting out your home through Airbnb? I saw this question repeated throughout many websites and queries. This is the question I wanted to answer and I chose to utilize data from one of Airbnb's most densely populated cities - London. 

&nbsp;
## Data

I retrieved this data set from [Inside Airbnb](http://insideairbnb.com/get-the-data.html). 

This data consisted of:
* 76,984 different listings
* 73 unique features
* 33 London boroughs

#### **Summarizing Price per Night**
* Mean: $112.11
* Std: $325.92
* Min: $0
* Max: $18,673.00
* 25%: $43.00
* 50%: $75.00
* 75%: $121.00

**98.4% of the Price per Night data fell below $500 per night**

&nbsp;
## Models

|**Model**|**Performance**|
------|------|
Linear Regression| RMSE: $52.21|
Random Forest| RMSE: $46.39|
XGBoost| RMSE: $46.66||

&nbsp;
## Limitations

The biggest limitation I see to the data that I am working with, is it taken at one stationary point in time. This does not factor in holiday or seasonal pricing trends or even changes in price or time. We can assume that price will gradually change with inflation and with trends in the London market due to business and many other factors. 

&nbsp;
## Tools & Packages

#### Stack:
* python
* git
* markdown

#### Modeling/Machine Learning
* Pandas
* Numpy
* Scikit-learn
* Linear Regression
* Random Forest Regressor
* XGB Regressor 

#### Data Visualization:
* Matplotlib
* Seaborn


&nbsp;
## Next Steps
1. Use web scraping to continually update data set to account for seasonal and holiday pricing trends.

2. Account for the very high priced listings.

3. Use Natural Language Processing to identify words used in listing titles or reviews that result in higher price. 

4. Build Dash app to build visualization
