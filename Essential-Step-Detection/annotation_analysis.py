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
            tmp_data['steps'].append(row['step_'+str(i+1)])
        tmp_data['process_name'] = row['process_name']


        annotations = list()
        for index, row in annotation_result.iterrows():
            if row['Answer.' + tmp_data['id'] + '_familiar.on'] in [True, False]:
                tmp_annotation = dict()
                tmp_annotation['familiar'] = row['Answer.' + tmp_data['id'] + '_familiar.on']
                tmp_annotation['essential_score_by_step'] = list()
                for i in range(num_steps):
                    tmp_annotation['essential_score_by_step'].append(row['Answer.' + tmp_data['id'] + '_s' + str(i+1)])
                tmp_annotation['missing_step'] = row['Answer.' + tmp_data['id'] + '_missing']
                tmp_annotation['missing_step_position'] = row['Answer.' + tmp_data['id'] + '_missing_position.label']
                annotations.append(tmp_annotation)
        result.append({'process_info': tmp_data, 'annotations': annotations})
        # print('lalala')
    return result


def analyze_annotation(num_steps):
    annotations = process_annotation(num_steps)
    familiar_count = 0
    annotation_result_count = {'-2': 0, '-1': 0, '0': 0, '1': 0, '2': 0}
    missing_step_count = 0
    for tmp_process in annotations:
        for tmp_a in tmp_process['annotations']:
            if tmp_a['familiar'] == True:
                familiar_count += 1
            for tmp_s in tmp_a['essential_score_by_step']:
                annotation_result_count[str(int(tmp_s))] += 1
            if tmp_a['missing_step'] not in ['no', 'good', 'NICE', 'essential']:
                missing_step_count += 1
    print('number of steps:', num_steps)
    print('number of familiar process:', familiar_count, '/', len(annotations)*5)
    print('annotation result:', annotation_result_count)
    print('percentage of essential events:', (annotation_result_count['2']+annotation_result_count['1'])/(annotation_result_count['-1']+annotation_result_count['-2']+annotation_result_count['0']+annotation_result_count['1']+annotation_result_count['2']))
    # print(missing_step_count)

    # We need to analyze the IAA
    num_pair = 0
    correct_pair = 0
    num_annotation = len(annotations[0]['annotations'])
    for tmp_annotation in annotations:
        for i in range(num_annotation):
            for j in range(num_annotation):
                if i != j:
                    for k in range(num_steps):
                        if tmp_annotation['annotations'][i]['essential_score_by_step'][k] in [1, 2]:
                            if tmp_annotation['annotations'][j]['essential_score_by_step'][k] in [1, 2]:
                                correct_pair += 1
                            num_pair += 1
                        elif tmp_annotation['annotations'][i]['essential_score_by_step'][k] in [-2, -1]:
                            if tmp_annotation['annotations'][j]['essential_score_by_step'][k] in [-2, -1]:
                                correct_pair += 1
                            num_pair += 1
                        # else:
                        #     if tmp_annotation['annotations'][j]['essential_score_by_step'][k] == 0:
                        #         correct_pair += 1
                        #     num_pair += 1
    print('agreement:', correct_pair, '/', num_pair, correct_pair/num_pair)




analyze_annotation(3)
analyze_annotation(4)
analyze_annotation(5)
analyze_annotation(6)
analyze_annotation(7)
analyze_annotation(8)
analyze_annotation(9)
analyze_annotation(10)


print('end')
