import pandas
import ujson as json
import random


non_essential_raw_data = pandas.read_csv('survey_pair_annotation/pair_annotation.csv')

id_to_annotation = dict()
id_to_pair = dict()

for index, row in non_essential_raw_data.iterrows():
    process_id = row['Input.pair_id']
    process = row['Input.process']
    step = row['Input.step']
    id_to_pair[process_id] = {'process': process, 'step': step}
    if process_id not in id_to_annotation:
        id_to_annotation[process_id] = list()
    id_to_annotation[process_id].append(row['Answer.' + str(process_id) + '_s1'])

final_data = list()

for tmp_id in id_to_pair:
    final_data.append({'process': id_to_pair[tmp_id]['process'], 'step': id_to_pair[tmp_id]['step'],
                       'annotation': id_to_annotation[tmp_id]})

# print(non_essential_raw_data)

with open('final_data/non-essential.json', 'w') as f:
    json.dump(final_data, f)


def process_annotation(num_steps):
    annotation_result = pandas.read_csv('annotation/ESD-' + str(num_steps) + '.csv')
    raw_original_data = pandas.read_csv('data/survey_data_' + str(num_steps) + '.csv')

    result = list()
    for index, row in raw_original_data.iterrows():
        tmp_data = dict()
        tmp_data['id'] = row['process_id']
        tmp_data['steps'] = list()
        for i in range(num_steps):
            tmp_data['steps'].append(row['step_' + str(i + 1)])
        tmp_data['process_name'] = row['process_name']

        annotations = list()
        for index, row in annotation_result.iterrows():
            if row['Answer.' + tmp_data['id'] + '_familiar.on'] in [True, False]:
                tmp_annotation = dict()
                tmp_annotation['familiar'] = row['Answer.' + tmp_data['id'] + '_familiar.on']
                tmp_annotation['essential_score_by_step'] = list()
                for i in range(num_steps):
                    tmp_annotation['essential_score_by_step'].append(
                        row['Answer.' + tmp_data['id'] + '_s' + str(i + 1)])
                tmp_annotation['missing_step'] = row['Answer.' + tmp_data['id'] + '_missing']
                tmp_annotation['missing_step_position'] = row['Answer.' + tmp_data['id'] + '_missing_position.label']
                annotations.append(tmp_annotation)
        result.append({'process_info': tmp_data, 'annotations': annotations})
        # print('lalala')
    return result


def analyze_annotation(num_steps):
    annotations = process_annotation(num_steps)
    pair_and_annotations = list()
    for tmp_process in annotations:
        for i in range(num_steps):
            tmp_process_name = str(tmp_process['process_info']['process_name'])
            tmp_step_name = str(tmp_process['process_info']['steps'][i])
            tmp_annotations = list()
            for tmp_a in tmp_process['annotations']:
                tmp_annotations.append(int(tmp_a['essential_score_by_step'][i]))
            pair_and_annotations.append({'process': tmp_process_name, 'step': tmp_step_name,
                                         'annotation': tmp_annotations})
    return pair_and_annotations


positive_pairs_3 = analyze_annotation(3)
positive_pairs_4 = analyze_annotation(4)
positive_pairs_5 = analyze_annotation(5)
positive_pairs_6 = analyze_annotation(6)
positive_pairs_7 = analyze_annotation(7)
positive_pairs_8 = analyze_annotation(8)
positive_pairs_9 = analyze_annotation(9)
positive_pairs_10 = analyze_annotation(10)

positive_pairs = positive_pairs_3 + positive_pairs_4 + positive_pairs_5 + positive_pairs_6 + positive_pairs_7 + positive_pairs_8 + positive_pairs_9 + positive_pairs_10


all_pairs = final_data + positive_pairs
random.shuffle(all_pairs)

with open('final_data/all_pairs.json', 'w') as f:
    json.dump(all_pairs, f)



print('end')
