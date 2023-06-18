from models.data.source import FamaFrenchData

data_path = r'C:\Users\serge\IdeaProjects\portfolio_manager\data\fama-french-factors'
filenames = ['Developed_5_Factors_Daily', 'Developed_MOM_Factor_Daily']

fm = FamaFrenchData(path_to_input_folder=data_path, filenames=filenames)
print(fm.data)