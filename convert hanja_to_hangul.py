import pandas as pd
import numpy as np
import re

def clean_text(s):
    hangul = re.compile('[^ ㄱ-ㅣ가-힣]+')
    result = hangul.findall(s)
    return result




train= pd.read_csv("train_data.csv")
test= pd.read_csv("test_data.csv")
submission = pd.read_csv("sample_submission.csv")
topic_dict = pd.read_csv("topic_dict.csv")

# remove special chracters
train["cleaned_title1"] = train["title"].str.replace(pat=r'[^\w]', repl=r' ', regex=True)
test ["cleaned_title1"] = test ["title"].str.replace(pat=r'[^\w]', repl=r' ', regex=True)


# replace hanja to hangul
train["cleaned_title2"] = train["cleaned_title1"].str.replace('美','미국 ')
test ["cleaned_title2"] = test ["cleaned_title1"].str.replace('美','미국 ')

train["cleaned_title2"] = train["cleaned_title2"].str.replace('韓','한국 ')
test ["cleaned_title2"] = test ["cleaned_title2"].str.replace('韓','한국 ')

train["cleaned_title2"] = train["cleaned_title2"].str.replace('日','일본 ')
test ["cleaned_title2"] = test ["cleaned_title2"].str.replace('日','일본 ')

train["cleaned_title2"] = train["cleaned_title2"].str.replace('獨','독일 ')
test ["cleaned_title2"] = test ["cleaned_title2"].str.replace('獨','독일 ')

train["cleaned_title2"] = train["cleaned_title2"].str.replace('靑','청와대 ')
test ["cleaned_title2"] = test ["cleaned_title2"].str.replace('靑','청와대 ')

train["cleaned_title2"] = train["cleaned_title2"].str.replace('北','북한 ')
test ["cleaned_title2"] = test ["cleaned_title2"].str.replace('北','북한 ')

train["cleaned_title2"] = train["cleaned_title2"].str.replace('英','영국 ')
test ["cleaned_title2"] = test ["cleaned_title2"].str.replace('英','영국 ')

train["cleaned_title2"] = train["cleaned_title2"].str.replace('中','중국 ')
test ["cleaned_title2"] = test ["cleaned_title2"].str.replace('中','중국 ')

train["cleaned_title2"] = train["cleaned_title2"].str.replace('伊','이탈리아 ')
test ["cleaned_title2"] = test ["cleaned_title2"].str.replace('伊','이탈리아 ')

train["cleaned_title2"] = train["cleaned_title2"].str.replace('UAE','아랍에미리트 ')
test ["cleaned_title2"] = test ["cleaned_title2"].str.replace('UAE','아랍에미리트 ')

train["cleaned_title2"] = train["cleaned_title2"].str.replace('EU','유럽 연합 ')
test ["cleaned_title2"] = test ["cleaned_title2"].str.replace('EU','유럽 연합 ')

train["cleaned_title2"] = train["cleaned_title2"].str.replace('與','여당 ')
test ["cleaned_title2"] = test ["cleaned_title2"].str.replace('與','여당 ')

train["cleaned_title2"] = train["cleaned_title2"].str.replace('軍','군대 ')
test ["cleaned_title2"] = test ["cleaned_title2"].str.replace('軍','군대 ')





train["uncleaned_title"] = train["cleaned_title2"].apply(lambda x : clean_text(x))
test["uncleaned_title"]  = test["cleaned_title2"].apply(lambda x : clean_text(x))







