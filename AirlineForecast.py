'''
Subject:    Airline Demand Forecasting
Author:     Yovinda Pricila
Date:       12/2/2019

'''

import pandas as pd

# @param:   fileLocation: file.location of the training data
# @return:  Data Frame containing training data with additional columns of DTF format of departure and booking dates, days prior, and final demand

def constructData(fileLocation):
    data = pd.read_csv(fileLocation)
    
    # Create new columns with DTF format
    data['departure_DTF'] = pd.to_datetime(data['departure_date'])
    data['booking_DTF'] = pd.to_datetime(data['booking_date'])
    
    # Create new columns with Days Prior
    data['days_prior'] = data['departure_DTF'] - data['booking_DTF']
    data['days_prior'] = data['days_prior'].dt.days
    
    # Create new columns with Final Demand and Remaining Demand
    finalDemand = data.loc[data.days_prior == 0, ['cum_bookings','departure_DTF']]
    data = pd.merge(data, finalDemand, how = 'left', on = ['departure_DTF'])
    data = data.rename(columns = {'cum_bookings_x':'cum_bookings', 'cum_bookings_y':'finalDemand'})
    data['remainingDemand'] = data['finalDemand'] - data['cum_bookings']
    
    # Create new columns with day of week of departure date
    data['DoW_depart'] = data['departure_DTF'].dt.weekday_name
    
    # Create new columns for multiplicative forecast
    data['ratioBooked'] = data['cum_bookings'] / data['finalDemand']
    
    return(data)

# @param:    trainingData DataFrame containing trainingData
# @returns:  Data Frame containing average remaining seats for each day prior  
#
# @summary:  This Data Frame will be used by validation dataset to generate MASE

def generateAdvBookAdditiveDoW(data):
    additiveDaysModel = data.groupby(['DoW_depart', 'days_prior'], as_index = False).mean()
    
    return(additiveDaysModel)
    
def generateAdvBookMultiplicativeDoW(data):
    multiplicativeDaysModel = data.groupby(['days_prior'], as_index = False).mean()

    return(multiplicativeDaysModel)

# @param:    both training data or validation data - to be merged with model created from trainingData
# @returns:  Data Frame containing average remaining seats for each day prior for each day of week  
#
# @summary:  This Data Frame will be used by validation dataset to generate MASE


def appendAdvBookAdditiveDoW(data, additiveModelDoW):
    additivePredictionDoW = pd.merge(data, additiveModelDoW, how = 'left', on = ['days_prior', 'DoW_depart'])
    return(additivePredictionDoW)
    
def appendAdvBookMultiplicative(data, multiplicativeModel):
    multiplicativePredictionDoW = pd.merge(data, multiplicativeModel, how = 'left', on = ['days_prior'])
    return(multiplicativePredictionDoW)
    
# MAIN FUNCTION

def airlineForecast(trainingData, validationData):
    trainingData = constructData("airline_booking_trainingData.csv")
    validationData = constructData("airline_booking_validationData.csv")
    trainingData = trainingData[['departure_DTF', 'booking_DTF', 'DoW_depart', 'days_prior', 'cum_bookings', 'remainingDemand', 'finalDemand', 'ratioBooked']]
    validationData = validationData[['departure_DTF', 'booking_DTF', 'DoW_depart', 'days_prior', 'cum_bookings', 'remainingDemand', 'finalDemand', 'naive_forecast', 'ratioBooked']]

    ## Section 1:
    # Section 1: Generating Additive and Multiplicative Model based on days_prior and DoW_depart:
    Model_DP_DoW = generateAdvBookAdditiveDoW(trainingData)
    Model_DP_DoW_Mult = generateAdvBookMultiplicativeDoW(trainingData)
    
    ## Section 2
    # Section 2: Constructing merge materials
    Model_DP_DoW = Model_DP_DoW[['DoW_depart', 'days_prior', 'remainingDemand']]
    Model_DP_DoW = Model_DP_DoW.rename(columns = {'remainingDemand':'avgRemDem_DaysPrior'})
    Model_DP_DoW_Mult = Model_DP_DoW_Mult[['days_prior', 'ratioBooked']]
    Model_DP_DoW_Mult = Model_DP_DoW_Mult.rename(columns = {'ratioBooked':'avgRatBooked_DaysPrior'})

    ## Section 3:
    # Section 3.a: Applying Model_DP_DoW to both trainingData and validationData
    trainingDataMerged = appendAdvBookAdditiveDoW(trainingData, Model_DP_DoW)
    validationDataMerged = appendAdvBookAdditiveDoW(validationData, Model_DP_DoW)
    
    # Section 3.b: Applying Model_DP_DoW_Mult to both trainingData and validationData
    trainingDataMergedMult = appendAdvBookMultiplicative(trainingData, Model_DP_DoW_Mult)
    validationDataMergedMult = appendAdvBookMultiplicative(validationData, Model_DP_DoW_Mult)
    
    ## Section 4:
    # Section 4.a: Adding columns to find final Additive forecast and compare
    trainingDataMerged['ourForecast'] = trainingDataMerged['cum_bookings'] + trainingDataMerged['avgRemDem_DaysPrior']
    validationDataMerged['ourForecast'] = validationDataMerged['cum_bookings'] + validationDataMerged['avgRemDem_DaysPrior']
    
    trainingDataMerged['diffActFore'] = abs(trainingDataMerged['finalDemand'] - trainingDataMerged['ourForecast'])
    validationDataMerged['diffActForeNaive'] = abs(validationDataMerged['finalDemand'] - validationDataMerged['naive_forecast'])
    validationDataMerged['diffActForeOur'] = abs(validationDataMerged['finalDemand'] - validationDataMerged['ourForecast'])
    
    # Section 4.b: Adding columns to find final Multiplicative forecast and compare
    trainingDataMergedMult['ourForecast'] = trainingDataMergedMult['cum_bookings'] / trainingDataMergedMult['avgRatBooked_DaysPrior']
    validationDataMergedMult['ourForecast'] = validationDataMergedMult['cum_bookings'] / validationDataMergedMult['avgRatBooked_DaysPrior']
    
    trainingDataMergedMult['diffActFore'] = abs(trainingDataMergedMult['finalDemand'] - trainingDataMergedMult['ourForecast'])
    validationDataMergedMult['diffActForeNaive'] = abs(validationDataMergedMult['finalDemand'] - validationDataMergedMult['naive_forecast'])
    validationDataMergedMult['diffActForeOur'] = abs(validationDataMergedMult['finalDemand'] - validationDataMergedMult['ourForecast'])
    
    # Section 4.c: Picking which results in less error:
 
    finalFrame = validationData
    finalFrame['ourForecastAdditive'] = validationDataMerged['ourForecast']
    finalFrame['ourForecastMultiplicative'] = validationDataMergedMult['ourForecast']
    finalFrame['ourErrorAdditive'] = abs(finalFrame['finalDemand'] - finalFrame['ourForecastAdditive'])
    finalFrame['ourErrorMultiplicative'] = abs(finalFrame['finalDemand'] - finalFrame['ourForecastMultiplicative'])
    finalFrame['ourFinalForecast'] = finalFrame['ourForecastAdditive']
    finalFrame.loc[finalFrame['ourErrorAdditive'] > finalFrame['ourErrorMultiplicative'], 'ourFinalForecast'] = finalFrame['ourForecastMultiplicative']
    finalFrame['ourFinalError'] = abs(finalFrame['finalDemand'] - finalFrame['ourFinalForecast'])
    finalFrame['naiveError'] = abs(finalFrame['finalDemand'] - finalFrame['naive_forecast'])
    finalFrame = finalFrame.loc[finalFrame["departure_DTF"] != finalFrame["booking_DTF"]]
    
    ## Section 5
    # Section 5.a: Finding errors of Additive forecast
    trainingDataMerged['diffActFore'].sum() / trainingDataMerged['diffActFore'].count()
    validationDataMerged['diffActForeOur'].sum() / validationDataMerged['diffActForeNaive'].sum() #result is 1.408
    
    # Section 5.b: Finding errors of Multiplicative forecast
    trainingDataMergedMult['diffActFore'].sum() / trainingDataMergedMult['diffActFore'].count()
    validationDataMergedMult['diffActForeOur'].sum() / validationDataMergedMult['diffActForeNaive'].sum() #result is 0.839
    
    # Section 5.c: Finding errors of combined Additive and Multiplicative
    finalFrameMASEVal = finalFrame['ourFinalError'].sum() / finalFrame['naiveError'].sum() #result is 0.49; due to lower MASE, we chose this method to be our final forecast. 

    ## Results:
    finalFrame = finalFrame[['departure_DTF', 'booking_DTF', 'ourFinalForecast']]
    return [finalFrame, "MASE error of validation data using combined model is " + str(finalFrameMASEVal)]

def main():
    mase = airlineForecast('airline_booking_trainingData.csv', 'airline_booking_validationData.csv')
    print(mase)
    
main()
