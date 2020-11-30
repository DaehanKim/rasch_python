from tqdm import tqdm
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

def parse_resp_as_label(resp):
    return int(float(resp)>0.5)

def parse_data(data_type = "skill_builder_only"):
    assert data_type in ('skill_builder_only', 'all'), "data_type should be either 'skill_builder_only' or 'all'"
    if data_type == 'skill_builder_only':
        with open('./ASSISTment_skill_builder_only_test_1123.txt','rt', encoding='utf8') as f:
            raw_data = f.readlines()
    else :
        with open('./assist_test_for_individual_model_1111.txt','rt', encoding='utf8') as f:
            raw_data = f.readlines()
        
    
    container = []
    triplet_data = []
    student_list = []
    whole_question_list = []
    for idx, line in tqdm(enumerate(raw_data), desc="preprocess data"):
        if idx %3 == 0:
            student = line.strip()
            container.append(student)
            student_list.append(student)
        elif idx %3 == 1:
            question_list = line.strip().split(',')
            container.append(question_list)
            whole_question_list.extend(question_list)
        elif idx %3 == 2 :
            response_list = line.strip().split(',')
            container.append(response_list)
        if len(container) >= 3 : 
            for ques, resp in zip(container[1], container[2]):
                triplet_data.append((container[0], ques, resp))
            container.clear()

    student_list = list(set(student_list))
    whole_question_list = list(set(whole_question_list))
    return student_list, whole_question_list, triplet_data


def test(dataset, irt_result):
    ''' dataset is a list of (student, question, response) '''
    stud_param, ques_param = irt_result
    pred = []
    true = []
    pass_cnt = 0
    for stud, ques, resp in dataset:
        try : 
            _prob = np.exp(stud_param[stud] - ques_param[ques])/(1+np.exp(stud_param[stud] - ques_param[ques]))
            pred.append(_prob)
            true.append(parse_resp_as_label(resp))
        except : 
            pass_cnt += 1
    
    pred, true = np.array(pred), np.array(true)

    acc = accuracy_score(true, (pred>0.5).astype(np.int))
    auc = roc_auc_score(true, pred)
    print(f"missed {pass_cnt} items due to key errors!")

    return acc, auc
            

def main():
    _, _, triplet_data = parse_data('all')
    with open('irt_result.pkl','rb') as f:
        irt_result = pickle.load(f)
    result = test(triplet_data, irt_result)
    print("Skill Builder and Non Skill Builder combined \nACC : {:.6f} | AUC : {:.6f}".format(*result))

if __name__ == "__main__" : 
    main()