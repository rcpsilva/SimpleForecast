from sklearn.ensemble import RandomForestRegressor
import util

def fit(df,forecasting_variable,
            steps_ahead=1,models=[RandomForestRegressor],
            variable_selection_type = 'Correlation',
            variable_lag_selection=False, hyperparameters=[],max_lags=30):

    """ Recieves multivariate time series data and returns a model

    Args:
        df: Data frame with multivariate time series data
        forecasting_variable: variable to be forecasted
        steps_ahead: Forescating horizon
        models: List of candidate models 
        variable_selection_type: Defines the variable selection method. Can be 'Correlation' or 'ExtraTrees'
        hyperparameters: List of dictionaries with hyper parameters for each model. It is used for 
        max_lags: Maximum number of lags in the model

    Return: 
        A fitted model for the time series
    
    """
    # Generates a new data frame with max_lags

    # Performs variable selection ()

    # Performs model selection ()

    return fitted_model


def variable_selection(df):
    """ Selects the best lags an the best exogenous variables for the given data
    
    """


def finetunnig(data, model, hyperparameters):
    """ Performs a grid search for the input model using the set of hyper parameters 
    
        Args:
            data: Data frame with multivariate time series data
            model: A scikit learn model
            hyperparameters: Set of candidate hyperparameters 

        Returns:
            best_model
            best_hyperparameters

    """
    
    return {'best_model': best_model,
            'best_hyperparameters': best_hyperparameters}

def test(df, model, metrics=['mae']):
    """ Evaluates a model given a test set
    
    """