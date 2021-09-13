#imports
import pandas as pd
import numpy as np
from datetime import timedelta
import statsmodels.api as sm
import time

def run_arimaDay(series, steps_ahead, configuracao):
  """ Receives a time series and returns the prediction for the specified horizon
    
        Args:
             series: Time series containing the date in year-month-day format and the variable for the forecast
             steps_ahead: Forescating horizon
             configuracao: Arima model parameters in the format (p,q,d)
             

        Returns:
            result: Predictions for the specified horizon
            t_fit: Time to do the training
            t_fcast: Time to make the prediction
                
            
  """
  result = []
  
  #Lista de data+hora que será previsto
  begin = series.index.max() + timedelta(days=0)
  date_list = [begin + timedelta(days=x) for x in range(1,steps_ahead+1)]
  
  #ores da série
  ues = series.ues

  #ARIMA
  start_fit = time.time()
  mod = sm.tsa.statespace.SARIMAX(ues, order=configuracao)
  res = mod.fit(disp=False)
  t_fit = time.time() - start_fit
  
  start_fcast = time.time() 
  forecast = res.forecast(steps=steps_ahead)
  t_fcast = time.time() - start_fcast 
  
  #Resultado no formato para ser exibido no gráfico
  for i in range(steps_ahead):
    if forecast[i] < 0: 
      result.append([date_list[i].strftime('%d/%m/%Y '),0])
    else:
      result.append([date_list[i].strftime('%d/%m/%Y '),round((forecast[i]),3)])

  return result, t_fit, t_fcast


def run_sarimaDay(series, steps_ahead, config_ordem, config_sazonal):
  """ Receives a time series and returns the prediction for the specified horizon
    
        Args:
             series: Time series containing the date in year-month-day format and the variable for the forecast
             steps_ahead: Forescating horizon
             config_ordem: Arima model parameters in the format (p,q,d)
             config_sazonal: Model parameters for the part with seasonality (P, Q, D)
             

        Returns:
            result:Predictions for the specified horizon
            t_fit: Time to do the training
            t_fcast:Time to make the prediction
                
            
  """
  result = []
  
  #Lista de data+hora que será previsto
  begin = series.index.max() + timedelta(days=0)
  date_list = [begin + timedelta(days=x) for x in range(1,steps_ahead+1)]
  
  #ores da série
  ues = series.ues

  #ARIMA

  start_fit = time.time()
  mod = sm.tsa.statespace.SARIMAX(ues, order=config_ordem, seasonal_order=config_sazonal)
  res = mod.fit(disp=False)
  t_fit = time.time() - start_fit
  
  start_fcast = time.time() 
  forecast = res.forecast(steps=steps_ahead)
  t_fcast = time.time() - start_fcast 

  #Resultado no formato para ser exibido no gráfico
  for i in range(steps_ahead):
    if forecast[i] < 0: 
      result.append([date_list[i],0])
    else:
      result.append([date_list[i],round((forecast[i]),3)])

  return result, t_fit, t_fcast

def run_sarimaxDay(series,exog,steps_ahead,config_ordem,config_sazonal):
  result = []
  
  #Lista de data+hora que será previsto
  begin = series.index.max() + timedelta(days=0)
  date_list = [begin + timedelta(days=x) for x in range(1,steps_ahead+1)]
  
  #Valores da série
  values = series.values
  
  #Valores da variável exogena
  ex = exog.values

  #Valores da variável exogena que será prevista
  ex_cast = ex.reshape(-1, 1)[-steps_ahead:]
  

  #ARIMA
  start_fit = time.time()
  mod = sm.tsa.statespace.SARIMAX(values, exog=ex, order=config_ordem, seasonal_order=config_sazonal)
  res = mod.fit(disp=False)
  t_fit = time.time() - start_fit

  start_fcast = time.time()
  forecast = res.forecast(steps=steps_ahead, exog=ex_cast)
  t_fcast = time.time() - start_fcast 
  #Resultado no formato para ser exibido no gráfico
  for i in range(steps_ahead):
    if forecast[i] < 0: 
      result.append([date_list[i].strftime('%d/%m/%Y %H:%M:%S'),0])
    else:
      result.append([date_list[i].strftime('%d/%m/%Y %H:%M:%S'),round((forecast[i]),3)])

  return result, t_fit, t_fcast