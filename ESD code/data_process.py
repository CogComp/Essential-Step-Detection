import csv
import json
'''
   read csv file and turn it into json
'''
all_pairs=[]
datalist = csv.reader(open('ESD.csv','rt',encoding='utf-8'))
count=0
for pair in datalist:
    if count!=0:
        tmp=dict()
        tmp['goal']=pair[0]
        tmp['step']=pair[1]
        tmp['label']=pair[5]
        tmp['id']=pair[6]
        all_pairs.append(tmp)
        print(tmp)
    count+=1


json_str=json.dumps(all_pairs)
with open('all_pairs.json','w') as json_file:
    json_file.write(json_str)
    


