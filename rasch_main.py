from tqdm import tqdm
from rasch import Rasch
import pickle

def main():
    with open('./assist_train_for_individual_model_1111.txt','rt', encoding='utf8') as f:
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
    model = Rasch(
        student_list = student_list,
        question_list=whole_question_list, 
        score_triplet_list= triplet_data)
    
    model.fit()
    params = model.get_params()
    # print(params)
    print(f"num student {len(student_list)} / num questions {len(whole_question_list)}")

    with open('irt_result.pkl','wb') as f:
        pickle.dump(params, f)
        
if __name__ == "__main__":
    main()