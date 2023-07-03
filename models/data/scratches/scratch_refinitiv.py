import eikon as ek
import pandas as pd
import time

path_apikeys = r'C:\Users\serge\OneDrive\Documents'
ek_api_key = pd.read_csv(fr'{path_apikeys}\apikeys.csv', names=['api', 'key'], index_col=0)
ek.set_app_key(ek_api_key.loc['reuters'].values[0])

ticker = 'MSFT.O'


fields = [
    'TR.CompanyMarketCap',
    'TR.TangibleBVPS',
    'TR.BasicWeightedAvgShares'
]



#
# df,err = ek.get_data(instruments=ticker,
#                  fields=fields)
# df_nasdaq, err = ek.get_data('0#.NDX', 'TR.RIC')
# df_nyse, err = ek.get_data('0#.NYA', 'TR.RIC')
# indices_rics = ['.DJI', '.STOXX', '.AEX', '.SBITOP', '.CRBEX', '.SOFIX', '.ATG']
# indices_rics = ['.SBITOP', '.CRBEX', '.SOFIX', '.ATG']
indices_rics = ['LVMH.PA', 'ASML.AS', 'NESN.S', 'LIN',
                'NOVOb.CO', 'AAPL.O', 'ROG.S', 'UNH',
                'SAPG.DE', 'MSFT.O']

indices_rics = ['.GDAXI']

print('Retrieving list of constituents')
all_rics = []
t0 = time.time()
for i in range(len(indices_rics)):
    temp_rics, err = ek.get_data(indices_rics[i], ['TR.IndexConstituentRIC' , 'TR.IndexConstituentName'], {'SDate':'20230321'})
    # all_rics, err = ek.get_data(indices_rics[2], ['TR.IndexConstituentRIC' , 'TR.IndexConstituentName'])

    print(f'Retrieved {time.time() - t0}')
    print(err)
    if err is None:
        all_rics.append(temp_rics['Constituent RIC'].to_list())



# df_dji_rics, err = ek.get_data('.DJI', ['TR.IndexConstituentRIC' , 'TR.IndexConstituentName'], {'SDate':'20230321'})
# df_dax, err = ek.get_data('.STOXX50E', ['TR.IndexConstituentRIC' , 'TR.IndexConstituentName'], {'SDate':'20230321'})
# df_italia_rics, err = ek.get_data('.FTITLMS', ['TR.IndexConstituentRIC' , 'TR.IndexConstituentName'], {'SDate':'20230321'})



# df_data, err = ek.get_data(all_rics, fields, {'SDate':'20230321'})
# print(df_data)
#
# df_dji, err = ek.get_data('.DJI', ['TR.IndexConstituentRIC' , 'TR.IndexConstituentName'], {'SDate':'20230321'})
#
# print('done')

ric = indices_rics[-1]

params = {'SDate':'1999-12-31',
          'EDate' :'2021-03-01',
          'Period': 'FQ0',
          'Frq':'FQ',
          'reportingState':'Rsdt',
          'curn':'Native',
          'Scale':'3',
          'RH': 'Date'}

res = {}
for ric in all_rics[0]:

    try:

        df = ek.get_data(ric, ['TR.BookValuePerShare', 'TR.BookValuePerShare.date'], parameters=params)
        res[ric] = df[0].iloc[:,1].values
        print(f'{ric}: Success!')
    except Exception as e:

        print(f'{ric}: {e}')



# interval = 'FY0'
# parameters = {'SDate' : '2018-01-01', 'EDate' : '2021-01-01', 'Frq' : interval}
ek.get_data('MSFT.O', fields={'TR.BookValuePerShare': { 'params':{'SDate' : '2010-01-01', 'EDate' : '2021-01-01'}}})
