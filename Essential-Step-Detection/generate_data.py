import ujson as json
import pandas


tmp_data = pandas.read_csv('data/1000.csv')
# print(tmp_data)


data_json = list()
for index, row in tmp_data.iterrows():
    if row['label'] == 1:
        data_json.append({'process': row['sent2'], 'steps': row['ending0']})

# print(len(data_json))


data_3 = list()
data_4 = list()
data_5 = list()
data_6 = list()
data_7 = list()
data_8 = list()
data_9 = list()
data_10 = list()
for i, tmp_sample in enumerate(data_json):
    tmp_process_name = tmp_sample['process'].split('How to ')[1]
    steps = tmp_sample['steps'].split('.')
    if steps[-1] == '':
        steps = steps[:-1]
    # print(len(steps))
    # if len(steps) not in len_count:
    #     len_count[len(steps)] = 0
    # len_count[len(steps)] += 1
    length = len(steps)
    if length == 3:
        tmp_row = dict()
        tmp_row['process_name'] = tmp_process_name
        tmp_row['process_id'] = '3_' + str(len(data_3))
        tmp_row['step_1'] = steps[0]
        tmp_row['step_2'] = steps[1]
        tmp_row['step_3'] = steps[2]
        data_3.append(tmp_row)
    elif length == 4:
        tmp_row = dict()
        tmp_row['process_name'] = tmp_process_name
        tmp_row['process_id'] = '4_' + str(len(data_4))
        tmp_row['step_1'] = steps[0]
        tmp_row['step_2'] = steps[1]
        tmp_row['step_3'] = steps[2]
        tmp_row['step_4'] = steps[3]
        data_4.append(tmp_row)
    elif length == 5:
        tmp_row = dict()
        tmp_row['process_name'] = tmp_process_name
        tmp_row['process_id'] = '5_' + str(len(data_5))
        tmp_row['step_1'] = steps[0]
        tmp_row['step_2'] = steps[1]
        tmp_row['step_3'] = steps[2]
        tmp_row['step_4'] = steps[3]
        tmp_row['step_5'] = steps[4]
        data_5.append(tmp_row)
    elif length == 6:
        tmp_row = dict()
        tmp_row['process_name'] = tmp_process_name
        tmp_row['process_id'] = '6_' + str(len(data_6))
        tmp_row['step_1'] = steps[0]
        tmp_row['step_2'] = steps[1]
        tmp_row['step_3'] = steps[2]
        tmp_row['step_4'] = steps[3]
        tmp_row['step_5'] = steps[4]
        tmp_row['step_6'] = steps[5]
        data_6.append(tmp_row)
    elif length == 7:
        tmp_row = dict()
        tmp_row['process_name'] = tmp_process_name
        tmp_row['process_id'] = '7_' + str(len(data_7))
        tmp_row['step_1'] = steps[0]
        tmp_row['step_2'] = steps[1]
        tmp_row['step_3'] = steps[2]
        tmp_row['step_4'] = steps[3]
        tmp_row['step_5'] = steps[4]
        tmp_row['step_6'] = steps[5]
        tmp_row['step_7'] = steps[6]
        data_7.append(tmp_row)
    elif length == 8:
        tmp_row = dict()
        tmp_row['process_name'] = tmp_process_name
        tmp_row['process_id'] = '8_' + str(len(data_8))
        tmp_row['step_1'] = steps[0]
        tmp_row['step_2'] = steps[1]
        tmp_row['step_3'] = steps[2]
        tmp_row['step_4'] = steps[3]
        tmp_row['step_5'] = steps[4]
        tmp_row['step_6'] = steps[5]
        tmp_row['step_7'] = steps[6]
        tmp_row['step_8'] = steps[7]
        data_8.append(tmp_row)
    elif length == 9:
        tmp_row = dict()
        tmp_row['process_name'] = tmp_process_name
        tmp_row['process_id'] = '9_' + str(len(data_9))
        tmp_row['step_1'] = steps[0]
        tmp_row['step_2'] = steps[1]
        tmp_row['step_3'] = steps[2]
        tmp_row['step_4'] = steps[3]
        tmp_row['step_5'] = steps[4]
        tmp_row['step_6'] = steps[5]
        tmp_row['step_7'] = steps[6]
        tmp_row['step_8'] = steps[7]
        tmp_row['step_9'] = steps[8]
        data_9.append(tmp_row)
    elif length == 10:
        tmp_row = dict()
        tmp_row['process_name'] = tmp_process_name
        tmp_row['process_id'] = '10_' + str(len(data_10))
        tmp_row['step_1'] = steps[0]
        tmp_row['step_2'] = steps[1]
        tmp_row['step_3'] = steps[2]
        tmp_row['step_4'] = steps[3]
        tmp_row['step_5'] = steps[4]
        tmp_row['step_6'] = steps[5]
        tmp_row['step_7'] = steps[6]
        tmp_row['step_8'] = steps[7]
        tmp_row['step_9'] = steps[8]
        tmp_row['step_10'] = steps[9]
        data_10.append(tmp_row)
    else:
        continue


prepared_data_3 = pandas.DataFrame(data_3)
prepared_data_3.to_csv('data/survey_data_3.csv', index=False)
# print(prepared_data_3)
prepared_data_4 = pandas.DataFrame(data_4)
prepared_data_4.to_csv('data/survey_data_4.csv', index=False)
prepared_data_5 = pandas.DataFrame(data_5)
prepared_data_5.to_csv('data/survey_data_5.csv', index=False)
prepared_data_6 = pandas.DataFrame(data_6)
prepared_data_6.to_csv('data/survey_data_6.csv', index=False)
prepared_data_7 = pandas.DataFrame(data_7)
prepared_data_7.to_csv('data/survey_data_7.csv', index=False)
prepared_data_8 = pandas.DataFrame(data_8)
prepared_data_8.to_csv('data/survey_data_8.csv', index=False)
prepared_data_9 = pandas.DataFrame(data_9)
prepared_data_9.to_csv('data/survey_data_9.csv', index=False)
prepared_data_10 = pandas.DataFrame(data_10)
prepared_data_10.to_csv('data/survey_data_10.csv', index=False)
