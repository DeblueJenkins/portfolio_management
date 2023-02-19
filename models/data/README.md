This module is based on the interaction between sources and variables. Sources are represent the raw data sources (APIs) and come in different shapes and forms. The idea is that each source class handles it's own data in its own way. Each source will have its own tables (currently pd.DataFrames) which will not be standardized. 
The Variables, however, unify this data in a standardized way. For example, PriceVariables will always need 
take as an input a table which contains the columns [High, Close, Open, Low, Adjusted Close] and produce a table
with columns [returns, adjusted_returns, high_minus_low, ...]. 

TO DO: Since API calls are restricted per time, it's difficult to debug while using APIs. Moreover, we would like to have a way 
to save the data once we have got it from the API. For this purpose, we will need 
one more class that will handle these tasks. That Source classes will initially save and/or
update data in SQL, while the some other class (call it for now DataHandler) will 
handle and load from these tables. This class should be standardized and able to work
with any other Source class.

