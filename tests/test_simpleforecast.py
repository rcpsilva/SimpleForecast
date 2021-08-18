
# import util
import pandas as pd
data_test = pd.read_csv("tests/test.csv") 



def soma_1(numero):
    return numero+1



def test_get_x2(a):

    # a=util.get_x2(data_test)
    # print(a)
    assert soma_1(41)== 42