import ujson as json
import pandas


tmp_data = pandas.read_csv('data/1000.csv')
print(tmp_data)


data_json = list()
for index, row in tmp_data.iterrows():
    if row['label'] == 1:
        data_json.append({'process': row['sent2'], 'steps': row['ending0']})


print('end')
