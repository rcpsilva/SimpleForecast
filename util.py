import numpy as np
import pandas as pd
import math
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from matplotlib import pyplot
from sklearn.model_selection import RandomizedSearchCV
from pprint import pprint
from sklearn.model_selection import GridSearchCV
import joblib



# def get_variable_list(): #List_format
#     pass

def column_Filter(df,steps_ahead,Target):
  """ Receives multivariate time series dataframe and returns a filtered list of steps ahead for later prediction
    
        Args:
             df: Data frame with multivariate time series data
             steps_ahead: Forescating horizon
             Target:String of target variable
             

        Returns:
            aux: list of lists containing
                 aux3[0]:lista (List of variables of interest for dataframe construction)
                 aux3[1]:lags (List of lags of interest for dataframe construction)
                 aux3[2]:list_Target ( List containing target variable)
                 aux3[3]:lags_list_Target ( List of lags for target variable)
            
  """
  list_column = []
  # pegando colunas do data frame
  for column in (df.columns):
    if column != 'Data':
      list_column.append(column)
  # prosseguindo com a filtração
  b=[]
  lags=[]
  aux3=[]
  lista=[]
  list_Target=[]
  lags_list_Target=[]
 
  # filtra e  remove colunas 
  for i in range(len(list_column)):
    j=list_column[i].split("_t-")
    if (int(j[1])>=steps_ahead):
      b.append(j)

  #localiza as colunas  e faz uma lista se a coluna for list_Target cria uma lista só pra ela
  for k in range(len(b)):
    if b[k][0] not in lista and b[k][0]!=Target:
        lista.append(b[k][0])
    elif b[k][0] not in list_Target and b[k][0]==Target:
        list_Target.append(b[k][0])
 
  # localiza as colunas e cria uma lista de lags para cada
  for i in range(len(lista)):
    aux=[]
    for k in range(len(b)):
      if (lista[i]==b[k][0])  :
          aux.append(int(b[k][1]))
    lags.append(aux)
    del(aux)
  
  if list_Target!=[]:
    for k in range(len(b)):
      if b[k][0]==list_Target[0]:
        lags_list_Target.append(int(b[k][1]))
 
  aux3.append(lista)
  aux3.append(lags)
  aux3.append(list_Target)
  aux3.append(lags_list_Target)
 

  return aux3
 
def resource_ranking(df,lista_filtrada3,Target,data_Itreino='1993-01-01',data_Ftreino='2011-12-31') :
  """ Receives multivariate time series dataframe and returns a dictionary containing the best features found using correlation and feature importance
    
        Args:
             df: Data frame with multivariate time series data
             steps_ahead: Forescating horizon
             Target: String of target variable
             data_Itreino: Start date for data training
             data_Ftreino: Last date for training
             lista_filtrada3
             

        Returns:
           A dictionary containing the best features found using correlation and feature importance
                        
  """
  tab=get_x2(df,lista_filtrada3[0],lista_filtrada3[1],lista_filtrada3[2],lista_filtrada3[3])
  train_selection = (tab[0]['Data'] >= data_Itreino) & (tab[0]['Data'] <= data_Ftreino)
  dff=tab[0][train_selection].drop("Data",axis=1)
  dff=dff.iloc[tab[1]:,:]
  array1 = dff.values
  df= df[train_selection].drop("Data",axis=1)
  df=df.iloc[tab[1]:,:]
  array2 = df[Target]
  X =array1[:,0:len(tab[0].columns)]
  Y = array2
  
  
  # feature extraction
  test = SelectKBest(score_func=f_regression, k=4)
  fit = test.fit(X, Y)
  # summarize scores
  set_printoptions(precision=3)
 

    
  # print("Selecao_univariada",fit.scores_)
  f=fit.scores_
  f_ord = sorted(f,reverse=True)
  # print("Selecao_univariada_ordenada",f_ord)

  ll=[]
  for y in range(len(f_ord)):
    for i in range(len(f)):
      if (f_ord[y]==f[i]):
        ll.append(i)
  leg_seq=[]
  for i in range(len(ll)):
    leg_seq.append(dff.columns[ll[i]])    

  # print("Colunas_selecionadas [0]- Selecao_univariada",leg_seq)
 
  model = ExtraTreesRegressor(n_estimators=10,random_state=42)
  model.fit(X,Y)
  g=model.feature_importances_
  g_ord=sorted(model.feature_importances_,reverse=True)
  # print("Importancia",g)
  # print("Importancia_ordenada",g_ord)
  jj=[]
  for y in range(len(g_ord)):
    for i in range(len(g)):
      if (g_ord[y]==g[i]):
        jj.append(i)
  leg_seq2=[]
  for i in range(len(ll)):
    leg_seq2.append(dff.columns[jj[i]])    
  # print("Colunas_selecionadas [1]- Importância do recurso",leg_seq2)
 
  return  {'Correlation': leg_seq,'feature_importances': leg_seq2}
    
def standardize_variable_list (resource_ranking,q,Target):
  """ Select the number of ranked attributes and standardize the format of attributes in lists
    
        Args:
             df: Data frame with multivariate time series data
             List:List of variables of interest for dataframe construction
             Target: Target variable
             q: number of ranked attributes.By default that= 10 best ranked attributes

        Returns:
               list of lists containing :
                 aux3[0]:lista (List of variables of interest for dataframe construction)
                 aux3[1]:lags (List of lags of interest for dataframe construction)
                 aux3[2]:list_Target ( List containing target variable)
                 aux3[3]:lags_list_Target ( List of lags for target variable)
            
  """
  b=[]
  lags=[]
  aux3=[]
  lista=[]
  list_Target=[]
  lags_list_Target=[]
  q = 10 if q==0 else q
  # filtra para valores referentes a quantidade
  for i in range(q):
    j=resource_ranking[i].split("_t-")
    b.append(j)
  print("lista_selecionada",b)

  #localiza as colunas  e faz uma lista se a coluna for list_Target cria uma lista só pra ela
  for k in range(len(b)):
    if b[k][0] not in lista and b[k][0]!=Target:
        lista.append(b[k][0])
    elif b[k][0] not in list_Target and b[k][0]==Target:
        list_Target.append(b[k][0])


 
  # localiza as colunas e cria uma lista de lags para cada
  for i in range(len(lista)):
    aux=[]
    for k in range(len(b)):
      if (lista[i]==b[k][0])  :
          aux.append(int(b[k][1]))
    lags.append(aux)
    del(aux)

  if list_Target!=[]:
    for k in range(len(b)):
      if b[k][0]==list_Target[0]:
        lags_list_Target.append(int(b[k][1]))
  print("lags_list_Target",lags_list_Target)
 
  aux3.append(lista)
  aux3.append(lags)
  aux3.append(list_Target)
  aux3.append(lags_list_Target)
 
  return aux3

def get_train_test_sets(df, list,lags,Target,lags_Target,data_Itest= '2013-01-01',data_Ftest= '2013-12-31',data_Itreino='1993-01-01',data_Ftreino='2011-12-31'):
  """ Groups data into training and testing subsets according to inferred date
    
        Args:
             df: Data frame with multivariate time series data
             List:List of variables of interest for dataframe construction
             Target: Target variable
             data_Itest= Start date for data testing
             data_Ftest= last test date
             data_Itreino=Start date for data training
             data_Ftreino=last date for training
             

        Returns:
           x1_train: List containing train data
           x1_test: List containing test data
           y1_train: List containing train data
           y1_test: List containing test data
            
 """
  
  
  
  tabela = get_x2(df, list,lags,Target,lags_Target)
  train_selection = (tabela[0]['Data'] >= data_Itreino) & (tabela[0]['Data'] <= data_Ftreino)
  test_selection = (tabela[0]['Data'] >= data_Itest) & (tabela[0]['Data'] <= data_Ftest)

  x1_train = tabela[0][train_selection].drop("Data", axis=1)
  x1_test = tabela[0][test_selection].drop("Data", axis=1)
  
  y1_train = df[Target][train_selection]
  y1_test = df[Target][test_selection]

  x1_train = x1_train[tabela[1]:]
  y1_train = y1_train[tabela[1]:]
 
           
  return x1_train, x1_test,y1_train, y1_test

def get_x2(df,list,lags,Target,lags_Target):
  """ Keeps the "Data" index and returns a DataFrame object with shifted index values.
    
        Args:
             df: Data frame with multivariate time series data
             list:list of variables of interest for dataframe construction
             lags:list of lags of interest for dataframe construction
             Target:list containing target variable
             lags_Target:list of lags for target variable

        Returns:
            dataX: Dataframe with shifted multivariate
            max_lag: Maximum number of lags in the new dataframe
            list: List of variables of interest for dataframe construction
            lags: List of lags of interest for dataframe construction
            Target: List containing target variable
            lags_Target: List of lags for target variable

  """
  lags=lags
  list_aux=[]
  list_aux2=[]
  max_lag=0;
  data = pd.DataFrame()
  dataX = pd.DataFrame()
  dataX['Data']=df['Data']
  for coluna in list:
    data[coluna] = df[coluna]       

  for i in range(len(lags)):
    for j in range(len(lags[i])):
      list_aux = data.iloc[:,i].tolist()
      for displacement in range((lags[i][j])):
        if max_lag<(lags[i][j]):
          max_lag=(lags[i][j])
        del list_aux[len(list_aux)-1]
        list_aux.insert(0,nan)
      dataX[((data.iloc[:,i]).name)+("_t-")+str(lags[i][j])]=(list_aux)  
  
  for i in range(len(lags_Target)):
    list_aux2=df[Target].iloc[:,0].tolist()
    for displacement in range((lags_Target[i])):
      if max_lag<(lags_Target[i]):
        max_lag=(lags_Target[i])
      del list_aux2[len(list_aux2)-1]
      list_aux2.insert(0,nan)
    dataX[((df[Target].iloc[:,0]).name)+("_t-")+str(lags_Target[i])]=list_aux2        
  return dataX,max_lag,list,lags,Target,lags_Target;


def get_x30(df,List, Target,steps_ahead=30):
  """ Keeps the "Data" index and returns a DataFrame object with index values ​​shifted by 30 time units
    
        Args:
             df: Data frame with multivariate time series data
             List:List of variables of interest for dataframe construction
             Target:List containing target variable
             steps_ahead: Forescating horizon.By definition 30 steps forward
             

        Returns:
            dataX: Dataframe with shifted multivariate by 30 time units
            max_lag: Maximum number of lags in the new dataframe
            List: List of variables of interest for dataframe construction
            lags: List of lags of interest for dataframe construction
            lags_Target: List of lags for target variable
            
 """
  ix = []
  idx =  [i for i in np.arange(1, steps_ahead+1)]
  for i in range(len(List)):
    ix.append(idx)

  dataX = get_x2(df, List, ix, Target, idx)
  return dataX
def get_column(df,Target):
  """ Keeps the "Data" index and returns a DataFrame object with shifted index values.
    
        Args:
             df: Data frame with multivariate time series data
             Target:Target variable

        Returns:
            A dictionary containing:
             Target:list containing target variable
             variable_list: List of variables of interest for dataframe construction
            

   """
  lista=[]
  list_Target=[]
  for column in (df.columns):
    if column != 'Data':
      if column != Target:
         lista.append(column)
      elif column==Target:
        list_Target.append(column)
        
      
  return {"variable_list":lista,"Target":list_Target}




# Models
def arvore(df,list, lags, Target, lags_Target,hyperparameters=[]):
  """ Receives multivariate time series dataframe and returns a error metric of selected model
    
        Args:
             df: Data frame with multivariate time series data
             list: List of variables of interest for dataframe construction
             lags: List of lags of interest for dataframe construction
             Target: List containing target variable
             lags_Target: List of lags for target variable
             hyperparameters:Parameter whose value is used to control the learning process

        Returns:
            list: List of variables of interest for dataframe construction
            lags: List of lags of interest for dataframe construction
            Target: List containing target variable
            lags_Target: List of lags for target variable
            rmse: Is the measure that calculates the "root mean square" of the errors between the observed (actual) values ​​and the predictions (assumptions)
            
  """
  
  x1_train, x1_test,y1_train, y1_test = get_train_test_sets(df, list,lags,Target,lags_Target)

  model = DecisionTreeRegressor(random_state = 42)
  
  model.fit(x1_train, y1_train)                 
  print("------------------")
  print("Arvore")
  y1_pred = model.predict(x1_test)
  mse = mean_squared_error(y1_test, y1_pred)
  std_mse=np.std(np.sqrt((y1_pred - y1_test)**2))
  
  print("std_mse",round(std_mse,2))
  rmse = math.sqrt(mse)
  print("Erro medio absoluto----",round(mean_absolute_error(y1_test, y1_pred),2))
  
  # pyplot.plot(np.arange(y1_test.shape[0]),y1_test, label='Expected tree ')
  # pyplot.plot(y1_pred, label='Predicted tree')
  # pyplot.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
  #               mode="expand", borderaxespad=0, ncol=3)
  # pyplot.show()
  return list,lags,Target,lags_Target,round(rmse,2)

def arvores(df,arvore_parametros,Target):
  """ Receives a list of lists with selected variables and corresponding steps forward returning error metrics for input lists
    
        Args:
             df: Data frame with multivariate time series data
             list: List of variables of interest for dataframe construction
             lags: List of lags of interest for dataframe construction
             Target: List containing target variable
             lags_Target: List of lags for target variable
             hyperparameters:Parameter whose value is used to control the learning process

        Returns:
            tb: dataframe with input data and error metrics
           
            
  """
  lista_colunas=["lista","lista_lags",Target,"lags_Target","rmse"]
  tb = pd.DataFrame(columns=lista_colunas)
  print("Arvores")
  for x in range(len(arvore_parametros)):
    a=arvore(df,arvore_parametros[x][0],arvore_parametros[x][1],arvore_parametros[x][2],arvore_parametros[x][3],variavel_Alvo)
 
    tb.loc[x,'lista']=a[0]
    tb.loc[x,'lista_lags']=a[1]
    tb.loc[x,Target]=a[2]
    tb.loc[x,'lags_Target']=a[3]
    tb.loc[x,"rmse"]=a[4]

  print(tb)

  return tb



def florestaAleatoria(df,list, lags, Target, lags_Target):
  """ Receives multivariate time series dataframe and returns a error metric of selected model
    
        Args:
             df: Data frame with multivariate time series data
             list: List of variables of interest for dataframe construction
             lags: List of lags of interest for dataframe construction
             Target: List containing target variable
             lags_Target: List of lags for target variable
             hyperparameters:Parameter whose value is used to control the learning process

        Returns:
            list: List of variables of interest for dataframe construction
            lags: List of lags of interest for dataframe construction
            Target: List containing target variable
            lags_Target: List of lags for target variable
            rmse: Is the measure that calculates the "root mean square" of the errors between the observed (actual) values ​​and the predictions (assumptions)
            
  """
  x1_train, x1_test,y1_train, y1_test = get_train_test_sets(df, list,lags,Target,lags_Target)
  model = RandomForestRegressor(random_state = 42 ,bootstrap=True, max_depth= 8, max_features= 5, min_samples_leaf= 4, min_samples_split= 8, n_estimators=1000)
  model.fit(x1_train, y1_train)
  print("------------------")
  print("floresta")
  y1_pred = model.predict(x1_test)
  mse = mean_squared_error(y1_test, y1_pred)
  std_mse=np.std(np.sqrt((y1_pred - y1_test)**2))
  rmse = math.sqrt(mse)
  print("Erro medio absoluto----",round(mean_absolute_error(y1_test, y1_pred),2))
  pyplot.plot(np.arange(y1_test.shape[0]),y1_test, label='Expected Random forests ')
  pyplot.plot(y1_pred, label='Predicted Random forests')
  pyplot.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=3)
  pyplot.show()
    
  return lista,lags,Eto,lags_eto,round(rmse,2),"RandomForestRegressor"

def florestasAleatorias(df,arvore_parametros,Target):
  """ Receives a list of lists with selected variables and corresponding steps forward returning error metrics for input lists
    
        Args:
             df: Data frame with multivariate time series data
             list: List of variables of interest for dataframe construction
             lags: List of lags of interest for dataframe construction
             Target: List containing target variable
             lags_Target: List of lags for target variable
             hyperparameters:Parameter whose value is used to control the learning process

        Returns:
            tb: dataframe with input data and error metrics
           
            
  """
  lista_colunas=["lista","lista_lags",Target,"lags_Target","rmse"]
  tb = pd.DataFrame(columns=lista_colunas)
  print("florestass")
  for x in range(len(arvore_parametros)):
    a=florestaAleatoria(df,arvore_parametros[x][0],arvore_parametros[x][1],arvore_parametros[x][2],arvore_parametros[x][3],variavel_Alvo)
    
    tb.loc[x,'model']=a[5]
    tb.loc[x,'lista']=a[0]
    tb.loc[x,'lista_lags']=a[1]
    tb.loc[x,'Eto']=a[2]
    tb.loc[x,'lags_eto']=a[3]
    tb.loc[x,"rmse"]=a[4]
    joblib.dump(a,"floresta")

  print(tb)

  return tb




def xgb(df,list, lags, Target, lags_Target):
  """ Receives multivariate time series dataframe and returns a error metric of selected model
    
        Args:
             df: Data frame with multivariate time series data
             list: List of variables of interest for dataframe construction
             lags: List of lags of interest for dataframe construction
             Target: List containing target variable
             lags_Target: List of lags for target variable
             hyperparameters:Parameter whose value is used to control the learning process

        Returns:
            list: List of variables of interest for dataframe construction
            lags: List of lags of interest for dataframe construction
            Target: List containing target variable
            lags_Target: List of lags for target variable
            rmse: Is the measure that calculates the "root mean square" of the errors between the observed (actual) values ​​and the predictions (assumptions)
            
  """
  
  x1_train, x1_test,y1_train, y1_test = get_train_test_sets(df, list,lags,Target,lags_Target)
  
  model = XGBRegressor( random_state = 42,colsample_bytree= 1, gamma= 5, learning_rate= 0.05, max_depth= 3, min_child_weight= 1, subsample= 1)
  model.fit(x1_train, y1_train)
  print("------------------")
  print("xgb")
  y1_pred = model.predict(x1_test)
  mse = mean_squared_error(y1_test, y1_pred)
  std_mse=np.std(np.sqrt((y1_pred - y1_test)**2))
  
  print("std_mse",round(std_mse,2))
  rmse = math.sqrt(mse)
  print("Erro medio absoluto----",round(mean_absolute_error(y1_test, y1_pred),2))
  # pyplot.plot(np.arange(y1_test.shape[0]),y1_test, label='Expected xgb ')
  # pyplot.plot(y1_pred, label='Predicted xgb')
  # pyplot.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
  #               mode="expand", borderaxespad=0, ncol=3)
  # pyplot.show()
    
  return list,lags,Target,lags_Target,round(rmse,2),"XGBRegressor"

def xgbs(df,arvore_parametros,Target):
  """ Receives a list of lists with selected variables and corresponding steps forward returning error metrics for input lists
    
        Args:
             df: Data frame with multivariate time series data
             list: List of variables of interest for dataframe construction
             lags: List of lags of interest for dataframe construction
             Target: List containing target variable
             lags_Target: List of lags for target variable
             hyperparameters:Parameter whose value is used to control the learning process

        Returns:
            tb: dataframe with input data and error metrics
           
            
  """
  lista_colunas=["lista","lista_lags",Target,"lags_Target","rmse"]
  tb = pd.DataFrame(columns=lista_colunas)
  print("xgbs")
  for x in range(len(arvore_parametros)):
    a=xgb(df,arvore_parametros[x][0],arvore_parametros[x][1],arvore_parametros[x][2],arvore_parametros[x][3],variavel_Alvo)
    
    tb.loc[x,'model']=a[5]
    tb.loc[x,'lista']=a[0]
    tb.loc[x,'lista_lags']=a[1]
    tb.loc[x,Target]=a[2]
    tb.loc[x,'lags_Target']=a[3]
    tb.loc[x,"rmse"]=a[4]

  print(tb)

  return tb



  #Grid

  
def gridSearchF(df, list,lags,Target,lags_Target):
  """ Performs a grid search on the manually specified subset of the target algorithm's hyperparameter space and returns the best hyperparameters
    
        Args:
             df: Data frame with multivariate time series data
             List:List of variables of interest for dataframe construction
             Target:List containing target variable
             steps_ahead: Forescating horizon.By definition 30 steps forward
             lags: List of lags of interest for dataframe construction
             lags_Target: List of lags for target variable
             List: List of variables of interest for dataframe construction
        Returns:
            grid_search.best_params_: Parameter setting that gave the best results on the hold out data.
            
            
            
            
 """
  x1_train, x1_test,y1_train, y1_test = get_train_test_sets(df, list,lags,Target,lags_Target)
   # Create the parameter grid based on the results of random search 
  param_grid = {
      'bootstrap': [True],
      'max_depth': [10,80, 90, 100, 110],
      'max_features': [2, 3],
      'min_samples_leaf': [3, 4, 5],
      'min_samples_split': [8, 10, 12],
      'n_estimators': [100, 200, 300, 1000]
  }

  # Create a base model
  rf = RandomForestRegressor(random_state = 42)

  # Instantiate the grid search model
  grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                            cv = 3, n_jobs = -1, verbose = 2, return_train_score=True)

  print(" -----------------grid_search.fit--------------")
 
  start_time=timer(None)

  grid_search.fit(x1_train, y1_train);

  timer(start_time)  
  
    
  return grid_search.best_params_

def gridSearchFs(df,parametros,Target):

  lista_colunas=["gridSearch"]
  tb = pd.DataFrame(columns=lista_colunas)
  
  print("gridSearchFs")
  for x in range(len(parametros)):
    a=gridSearchF(df,parametros[x][0],parametros[x][1],parametros[x][2],parametros[x][3],Target)
 
    tb.loc[x,'Best_ramdon_hyperparameter']=[a]
  print('---------------------------------------------------------------------------------')
  

  return tb

# gridSearch Arvore

def gridSearchA(df, list,lags,Target,lags_Target):
  """ Performs a grid search on the manually specified subset of the target algorithm's hyperparameter space and returns the best hyperparameters
    
        Args:
             df: Data frame with multivariate time series data
             List:List of variables of interest for dataframe construction
             Target:List containing target variable
             steps_ahead: Forescating horizon.By definition 30 steps forward
             lags: List of lags of interest for dataframe construction
             lags_Target: List of lags for target variable
             List: List of variables of interest for dataframe construction
        Returns:
            grid_search.best_params_: Parameter setting that gave the best results on the hold out data.  
 """
  tb = pd.DataFrame()
  x1_train, x1_test,y1_train, y1_test = get_train_test_sets(df, list,lags,Target,lags_Target)
   # Create the parameter grid based on the results of random search 
  
  param_grid={"splitter":["best","random"],
            "max_depth" : [1,3,5,7,9,11,12],
            "min_samples_leaf":[1,2,3,4,5,6,7,8,9,10],
            "min_weight_fraction_leaf":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
            "max_features":["auto","log2","sqrt",None],
            "max_leaf_nodes":[None,10,20,30,40,50,60,70,80,90] 
            }
 
  # Create a base model
  rf = DecisionTreeRegressor(random_state = 42)

  # Instantiate the grid search model
  grid_search=GridSearchCV(rf,param_grid=param_grid,scoring='neg_mean_squared_error',cv=3,verbose=3,n_jobs = -1,return_train_score=True)
  

  print(" -----------------grid_search.fit--------------")
  start_time=timer(None)

  grid_search.fit(x1_train, y1_train);

  timer(start_time)  

  return grid_search.best_params_

def gridSearchAs(df,parametros,Target):

  print('---------------------------------------------------------------------------------')
  
  for x in range(len(parametros)):
    a=gridSearchA(df,parametros[x][0],parametros[x][1],parametros[x][2],parametros[x][3],Target)
    
    print(x,"Best_hyperparameter_Arvores", a)
   
  print('---------------------------------------------------------------------------------')
  

  return tb

def gridSearchXgb(df, lista,lags,Eto,lags_eto,variavel_Alvo):
  """ Performs a grid search on the manually specified subset of the target algorithm's hyperparameter space and returns the best hyperparameters
    
        Args:
             df: Data frame with multivariate time series data
             List:List of variables of interest for dataframe construction
             Target:List containing target variable
             steps_ahead: Forescating horizon.By definition 30 steps forward
             lags: List of lags of interest for dataframe construction
             lags_Target: List of lags for target variable
             List: List of variables of interest for dataframe construction
        Returns:
            grid_search.best_params_: Parameter setting that gave the best results on the hold out data.  
 """
  x1_train, x1_test,y1_train, y1_test = train_test(df, lista,lags,Eto,lags_eto,variavel_Alvo)
   # Create the parameter grid based on the results of random search 
  param_grid = {
        'min_child_weight': [1, 5, 10],
        # Define a soma mínima dos pesos de todas as observações exigidas em uma criança.
        # Isso é semelhante a min_child_leaf no GBM, mas não exatamente. Isso se refere à “soma dos pesos” mínimas das observações, enquanto o GBM tem “número mínimo de observações”.
        # Usado para controlar o sobreajuste.
        'gamma': [0.5, 1, 1.5, 2, 5],
        # Um nó é dividido apenas quando a divisão resultante dá uma redução positiva na função de perda.
        # Gama especifica a redução de perda mínima necessária para fazer uma divisão.
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        # Semelhante a max_features em GBM. Denota a fração de colunas a serem amostras 
        # aleatoriamente para cada árvore.Valores típicos: 0,5-1
        'max_depth': [3, 4, 5],
        # booster [default = gbtree]
        # Selecione o tipo de modelo a ser executado em cada iteração. Possui 2 opções:
        # gbtree: modelos baseados em árvore
        # gblinear: modelos lineares
        }

  # Create a base model
  rf = XGBRegressor(random_state = 42)

  # Instantiate the grid search model
  grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                            cv = 3, n_jobs = -1, verbose = 2, return_train_score=True)

  print(" -----------------grid_search___Xgb.fit--------------")
  grid_search.fit(x1_train, y1_train);
    
  return grid_search.best_params_

def gridSearchXgbs(df,parametros,variavel_Alvo):
  tb = pd.DataFrame()
  for x in range(len(parametros)):
    a=gridSearchXgb(df,parametros[x][0],parametros[x][1],parametros[x][2],parametros[x][3],variavel_Alvo)
 
    tb.loc[x,'Best_hyperparameter_Xgb']=[a]
    print(x,"--->",a)

  print('---------------------------------------------------------------------------------')
  return tb

def gridSearch_models(df, lista,lags,Eto,lags_eto,variavel_Alvo):
  models = {
    'DecisionTreeRegressor': DecisionTreeRegressor(),
    'RandomForestRegressor': RandomForestRegressor(),
    'XGBRegressor': XGBRegressor()
  }


  params= {
    'DecisionTreeRegressor':{"splitter":["best"],
            "max_depth" : [1,3,5,8],
            "min_samples_leaf":[1,2,3,4],
            'n_estimators': [100, 500, 1000]
            },
    'RandomForestRegressor':  {
      "max_depth" : [1,3,5,8],
     "min_samples_leaf":[1,2,3,4],
     'min_samples_split': [8, 10, 12],
      'n_estimators': [100, 500, 1000]
      },

    'XGBRegressor':{
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5],
        }

    }

  for name in models.keys():
    est = models[name]
    est_params = params[name]
 return best_model,bestparamet