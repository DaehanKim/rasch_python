import torch.nn as nn
import torch 
from tqdm import tqdm
import sys
import visdom as vis
import numpy as np

NUM_EPOCH = 10
LEARNING_RATE = 0.0007
BATCH_SIZE = 64

def parse_resp(resp):
    return float(resp)

class Loader:
    def __init__(self, student_list, question_list, score_triplet_list):
        self.student_list = student_list
        self.question_list = question_list
        self.stud2id = {item:idx for idx, item in enumerate(student_list)}
        self.ques2id = {item:idx for idx, item in enumerate(question_list)}
        
        # convert to id
        self.score_triplet_list = [(self.stud2id[stud], self.ques2id[ques], parse_resp(resp)) for stud,ques,resp in tqdm(score_triplet_list, desc="loading batch loader")]
        self.loader = torch.utils.data.DataLoader(self.score_triplet_list, batch_size = BATCH_SIZE, shuffle=True)
    
    # def get_batch(self):
    #     for batch in self.loader:
    #         yield batch


class Rasch(nn.Module):
    def __init__(
        self,
        student_list : list, # list of student_i
        question_list : list, # list of item_j
        score_triplet_list : list # list of (student_i,student_j,0 or 1)
        ):
        super(Rasch, self).__init__()
        self.score_triplet_list = score_triplet_list 
        self.student_list = student_list
        self.question_list = question_list
        self.stud2id = {item:idx for idx, item in enumerate(student_list)}
        self.ques2id = {item:idx for idx, item in enumerate(question_list)}

        self.num_epoch = NUM_EPOCH
        self.params = nn.Embedding(len(student_list) + len(question_list), 1)
        self.loader = Loader(student_list=student_list, question_list=question_list, score_triplet_list=score_triplet_list)
        # self.params = nn.Parameter(torch.zeros(len(student_list) + len(question_list)))
        self.optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=LEARNING_RATE, 
            weight_decay=.0)
        self.loss_monitor = {}
        self.vis = vis.Visdom()
        self.plot = None
        self.last_batch_num = 0

    def fit(self):
        num_batch = int(len(self.score_triplet_list)/BATCH_SIZE)+1
        for epoch in range(self.num_epoch):
            self.loss_monitor[epoch + 1] = []
            for idx, batch in enumerate(self.loader.loader):
                loss = self.forward(batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.loss_monitor[epoch+1].append(loss.item())
                sys.stdout.write('EPOCH {} | {} / {} | nll loss : {:.4f}\r'.format(epoch + 1, idx+1, num_batch, loss.item()))
            print("EPOCH {} | {} / {} | nll loss : {:.4f}\r".format(epoch + 1, idx+1, num_batch, loss.item()))
            self.plot_loss_progress(epoch+1)
            
        print('Finished Training with epoch {}'.format(epoch + 1))
                     

    def forward(self, batch):
        stud_param = self.params(batch[0])
        ques_param = self.params(len(self.student_list)+batch[1])
        
        loglike = batch[2] * (stud_param - ques_param)
        loglike -= torch.log(1+torch.exp(stud_param - ques_param))
        return -loglike.mean() # negative log likelihood 

    def plot_loss_progress(self, epoch):
        
        # for epoch in range(1,len(self.loss_monitor)+1):
        xrange = np.array(range(self.last_batch_num,self.last_batch_num + len(self.loss_monitor[epoch])))
        if self.plot is None :
            self.plot = self.vis.line(Y = self.loss_monitor[epoch], X = xrange)
        else:
            self.vis.line(Y = self.loss_monitor[epoch], X = xrange, update='append', win=self.plot)
        self.last_batch_num += len(self.loss_monitor[epoch])
                
    def get_params(self):
        params = self.params.weight.detach().numpy()
        student_params = {stud : params[idx][0] for idx, stud in enumerate(self.student_list)}
        question_params = {ques : params[len(self.student_list)+idx][0] for idx, ques in enumerate(self.question_list)}
        return student_params, question_params