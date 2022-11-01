import pandas as pd
import os

dir_name = os.path.abspath('../../../Downloads')

if os.path.exists(dir_name):
    file_name = os.listdir(dir_name)[1]
    full_path = os.path.join(dir_name, file_name)

    dtf = pd.read_excel(full_path)
    dtf.to_csv(os.path.join(dir_name, 'Images.csv'))
