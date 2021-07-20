import json
import pandas as pd

with open('ynat-v1_dev.json','r',encoding='utf-8') as f:
    contents = f.read()
    json_data = json.loads(contents)
    
df = pd.DataFrame( columns=['title','topic_idx'])

for i in range(len(json_data)):
    sample_list = [ (json_data[i]['title'],json_data[i]['label'])]
    sample = pd.DataFrame(sample_list,columns=['title','topic_idx'])
    df=df.append(sample,ignore_index=True)

path = "C:/Users/PC/Desktop/KLUE_dev.csv"
df.to_csv(path, index = False)