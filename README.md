## Table of contents
* [Introduction](#general-info)
* [Technologies](#technologies)
* [Setup](#setup)
* [Files](#attached-files)

## Introduction
This project is analyzing the impact of human mobility, unemployment, and weather on Covid-19  
## Technologies
The project is implemented using:
* Python 3.8 (64-bit)
* KMean Clustering

## Setup
To run this project, you have to install:
* Anconda Navigator: it will include all the required libraries and just run it.
 
## Files
* Weather-dataset.py-----> Extract Weather Information worldwide from the nearest station [NOTICE: TAKES LONG TIME TO COLLECT DATA WORLDWIDE FROM THE NEAREST STATION]
* main.py --------> Extract Covid-19 cases (Confirmed, Recovered, Deaths) Dataset, Google Mobility Trends Dataset, Weather dataset 
                    Get Correlation of Covid-19 cases with both mobility and weather for overall countries and for each country individually.
*  Unemployment.py-----> Extract unemployment rate dataset, Get Correlation between Covid-19 cases and unemployment rate for overall countries and for each country individually.
* Cluster_mobility_covid.py -----> Construct K-means clustering algorithm for Covid-19 cases with respect to Grocery, Parks, Retail Mobility trends. In addition, Silhouette                                          Analysis is performed for different number of clusters 2, 3, 4, 5, and 6
* Cluster_unemployment_covid.py ------> Construct K-means clustering algorithm for Covid-19 cases with respect to Unemployment Rate. In addition, Silhouette                                                               Analysis is performed for different number of clusters 2, 3, 4, 5, and 6
* Cluster_weather_covid.py ----------> Construct K-means clustering algorithm for Covid-19 cases with respect to Weather. In addition, Silhouette                                                                          Analysis is performed for different number of clusters 2, 3, 4, 5, and 6
