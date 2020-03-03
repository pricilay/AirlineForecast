# AirlineForecast
This project is written in Python - utilizing Pandas and Numpy for data manipulation. 

Demand forecasting is a crucial part of airline revenue management that aims to maximize revenueby  matching  demand  to  available  capacity.   Especially  short-term  demand  forecasts  of  4  weeksor less that precede the target departure day have critical impact on airline revenue managementoperations such as pricing decisions and inventory control. In short-term forecasting, one of the mostimportant data is advance booking information.  Airline customers may request reservations days,weeks, or even months prior to their intended departure date.  Airlines maintain these reservationprofiles  for  each  calender  day,  which  is  partial  data  until  the  departure  date.   Advance  bookinginformation, while incomplete, reflects the most recent demand shifts and seasonality.

This analysis was done by conducting each empirical approach individually and comparing the MASE and MAD of each model. The additive approach is to sum the cumulative booking and average of remaining seats based on days-prior, whereas the multiplicative approach is to divide the cumulative booking by booking rate based on days-prior.

DATA

I was provided with two sets of airline booking data for a flight: the one for model estimation (test) and the other for validation. Following are the given data columns: 

•departure date:  departure date of the flight.
•booking date:  date in which booking requests arrived.
•cumulative  bookings:   number  of  cumulative  bookings  for  the  given  departure  date. The number of cumulative bookings on days prior =0 (i.e.  when departure date = booking date)is the final demand.
•final demand (validation data set only):  final demand of the given departure date.
•naive forecasts (validation data set only):  naive forecast of final demand for the given depar-ture date.

OBJECTIVE 

Write a function “airlineForecast(trainingDataFile,validationDataFile)”, which takes in two data: 
•input:  file names for training and validation data set.
•output:  a list of following results
   – MASE (Mean Absolute Scaled Error) 
   – forecasts:  a dataFrame which contains departure date, booking date, and forecasts fromyour model. 
   
APPROACH 

Combining both additive and simple multiplicative method to find the lowest error. By means, we compare the final forecast of the modified additive and simple multiplicative method on the training data, and picked the final forecast which yielded the lowest error.
   
RESULT 

final MASE error is lowered to 49%; reducing error level by 51 percentage point. 

