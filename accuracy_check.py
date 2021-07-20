import pandas as pd

#pred for prediction result array

train= pd.read_csv("train_data.csv")
test= pd.read_csv("test_data.csv")
submission = pd.read_csv("sample_submission.csv")
topic_dict = pd.read_csv("topic_dict.csv")
answer = pd.read_csv("answer.csv")

result = pd.DataFrame( columns=['title','pred','answer'])
result['title'] = test['title']
result['pred'] = pred
result['answer'] = answer['topic_idx']

wrong_classification=pd.DataFrame( columns=['title','pred','answer'])

for i in range(len(result)):
    if (result.iloc[i]['answer']!=result.iloc[i]['pred']):
        wrong_classification = wrong_classification.append(result.loc[i])

accuracy =round( (len(result)-len(wrong_classification))/len(result) *100,4)

print("accuracy :"+str(accuracy)+"%")