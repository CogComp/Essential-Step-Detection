import ujson as json
import pandas

# with open('HieVe_data/event.complex.json', 'r') as f:
#     tmp_data = json.load(f)
#
# selected_processes = list()
# for tmp_id in tmp_data:
#     for tmp_key in tmp_data[tmp_id]:
#         if len(tmp_data[tmp_id][tmp_key]) == 3:
#             selected_processes.append(
#                 {'process': tmp_key, 'step_1': tmp_data[tmp_id][tmp_key][0], 'step_2': tmp_data[tmp_id][tmp_key][1],
#                  'step_3': tmp_data[tmp_id][tmp_key][2], 'Is Step 1 a valid subevent?': 0, 'Is Step 2 a valid subevent?': 0, 'Is Step 3 a valid subevent?': 0, 'Is Step 1 Essential?': 0, 'Is Step 2 Essential?': 0, 'Is Step 3 Essential?': 0})
#
# result = pandas.DataFrame(selected_processes)
# result.to_csv('HieVe_data/HieVe_annotation.csv')
# print(result)

print('end')
