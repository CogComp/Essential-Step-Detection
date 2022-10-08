import pandas


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
    annotation_result = pandas.read_csv('annotation_adding_non_essential/non-ESD-' + str(num_steps) + '.csv')
    raw_original_data = pandas.read_csv('data/survey_data_' + str(num_steps) + '.csv')

    # print(annotation_result)
    process_step_pairs = list()
    for index, row in raw_original_data.iterrows():
        tmp_data = dict()
        tmp_data['id'] = row['process_id']
        tmp_data['process_name'] = row['process_name']
        # print(row)

        for anno_index, anno_row in annotation_result.iterrows():
            if type(anno_row['Answer.' + tmp_data['id'] + '_missing']) == str:
                process_step_pairs.append(
                    {'process': tmp_data['process_name'].replace('\n', ' ').replace('\r', ' '), 'step': anno_row['Answer.' + tmp_data['id'] + '_missing'].replace('\n', ' ').replace('\r', ' ')})
                # print(anno_row['Answer.' + tmp_data['id'] + '_missing'] != 'nan')
                # print(anno_row['Answer.' + tmp_data['id'] + '_missing'])
                # print(type(anno_row['Answer.' + tmp_data['id'] + '_missing']))
    print('number of collected pairs:', len(process_step_pairs))
    print(process_step_pairs)
    return process_step_pairs


annotation_3 = analyze_annotation(3)
annotation_4 = analyze_annotation(4)
annotation_5 = analyze_annotation(5)
annotation_6 = analyze_annotation(6)
annotation_7 = analyze_annotation(7)
annotation_8 = analyze_annotation(8)
annotation_9 = analyze_annotation(9)
annotation_10 = analyze_annotation(10)

all_annotations = annotation_3 + annotation_4 + annotation_5 + annotation_6 + annotation_6 + annotation_7 + annotation_8 + annotation_9 + annotation_10

pairs_for_annotation = list()
test_pairs = list()
for i, tmp_pair in enumerate(all_annotations):
    tmp_pair['pair_id'] = i
    pairs_for_annotation.append(tmp_pair)
    if i < 5:
        test_pairs.append(tmp_pair)

pairs_for_annotation = pandas.DataFrame(pairs_for_annotation)
pairs_for_annotation.to_csv('data/pairs_for_annotation.csv', index=False)

test_pairs = pandas.DataFrame(test_pairs)
print(test_pairs)
test_pairs.to_csv('data/test_pairs.csv', index=False)

print('end')
