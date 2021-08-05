import pandas as pd
from simpleforecast import fit


path= 'C:/Users/Ray/Documents/GitHub/'
data_Patricia_Eto = pd.read_csv(path+"SimpleForecast/data_PEto.csv") 
data_Patricia_Eto = pd.read_csv(path+"SimpleForecast/data_PEto.csv") 
print(data_Patricia_Eto)



auxiliary_variables=['Tmax', 'Tmin', 'I', 'Tmean', 'UR', 'V', 'J']
resultado = fit(data_Patricia_Eto,"Tmean",10,7,0,'Correlation',max_lags=30)

print(resultado)