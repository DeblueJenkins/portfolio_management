from models.data.source import Eikon



PATH_API_KEYS = r'C:\Users\serge\OneDrive\reuters\apikeys.csv'
api = Eikon(PATH_API_KEYS)

START_DATE = '1990-12-31'
END_DATE = '2024-04-27'

indices = ['.SPX', '.NDX', '.DJI']
res = {}
for ind in indices:
    res[ind] = api.get_index_constituents(index=ind, date='20240427')

res_ls = sum([v for k,v in res.items()], [])



for a in res_ls:
    print(f"{a}:")
    print(f"  constr: {0.05}")

