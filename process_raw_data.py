import pandas as pd 
import numpy as np 
from tqdm import tqdm
import sys 
from datetime import datetime as dt

SKILL_BUILDER_DATA = 'skill_builder_data_corrected_collapsed.csv'
NON_SKILL_BUILDER_DATA = 'non_skill_builder_data_new.csv'
TEST_RATIO = 0.25


dtype_dict = {
    'order_id':str,
    'user_id':str,
    'problem_id':str,
    'skill_id':str,
    'correct':str
}

def check_data(data_df):
    '''This function helps debug dataframe'''
    na_cnt = data_df.isna().sum(axis=0)
    print(na_cnt)
    print(data_df.head())
    # print(data_df.count(axis=1))

def get_standard_split(data_dict):
    train_data_dict, test_data_dict = {}, {}
    for user_id, data in tqdm(data_dict.items(), desc='Getting train/test split'):
        # if data is too small, whole data belongs to training set
        test_idx = len(data) if len(data) < 1/TEST_RATIO else int(len(data)*(1-TEST_RATIO))
        train_data_dict[user_id] = data[:test_idx]
        if test_idx != len(data) : test_data_dict[user_id] = data[test_idx:]
    return train_data_dict, test_data_dict


def get_txt_from_data_dict(data_dict):
    txt = ''
    container = []
    for user_id, data in data_dict.items():
        container.append(user_id)
        container.append(",".join([e[0] for e in data]))
        container.append(",".join([e[1] for e in data]))
        txt += "\n".join(container)
        txt += "\n"
        container.clear()
    return txt

if __name__ == "__main__":
    assert sys.argv[1] in ('skill_builder_only','all'), "Options must be either 'skill_builder_only' or 'all'"
    option = sys.argv[1]

    skill_df = pd.read_csv(SKILL_BUILDER_DATA, header=0, index_col=None, encoding='utf8', dtype=dtype_dict, usecols = ['order_id', 'user_id', 'problem_id', 'skill_id','correct'])
    nskill_df = pd.read_csv(NON_SKILL_BUILDER_DATA, header=0, encoding='utf8', dtype=dtype_dict, usecols = ['order_id', 'user_id', 'problem_id', 'skill_id', 'correct'])
    skill_df['skill_id'] = [','.join(str(item).split('_')) for item in skill_df['skill_id']]
    skill_df = skill_df[~skill_df['skill_id'].apply(pd.to_numeric, errors='coerce').isna()]

    if option == 'skill_builder_only':
        assist_df = skill_df
    else :     
        assist_df = pd.concat([nskill_df, skill_df], axis=0)
    assist_df = assist_df.sort_values(by=['order_id','user_id','problem_id'])

    data_dict = {} 
    for idx, row in tqdm(assist_df.iterrows(), desc="Dataframe into Dict", total=assist_df.shape[0]): 
        skills = row['skill_id'].split(',')
        user_id = row['user_id']
        correct = row['correct']
        if user_id not in data_dict : data_dict[user_id] = []
        data_dict[user_id].extend([(sk, correct) for sk in skills])
    
    train_dict, test_dict = get_standard_split(data_dict)
    timestamp = dt.now()
    with open('ASSISTment_{}_train_{:02d}{:02d}.txt'.format(option, timestamp.month, timestamp.day),'w',encoding='utf8') as f:
        train_txt = get_txt_from_data_dict(train_dict)
        f.write(train_txt)
    with open('ASSISTment_{}_test_{:02d}{:02d}.txt'.format(option, timestamp.month, timestamp.day),'w',encoding='utf8') as f:
        test_txt = get_txt_from_data_dict(test_dict)
        f.write(test_txt)


    

