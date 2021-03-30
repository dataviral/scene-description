import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import os
from . import global_vars


class SceneDescriptionDataset(torch.utils.data.Dataset):

    def __init__(self, data, mappings, imgs_path):
        self.data = data
        self.mappings = mappings
        self.imgs_path = imgs_path
        self.trainTransform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        question_id, answer_id, answer_vqa_id, img_num, map_name = self.data[i]
        question_id, answer_id = str(question_id), str(answer_id)
        if map_name != "sparky": img_num = str(int(img_num) + 1) # fix the image num

        question = self.mappings["qid_2_question"][question_id]
        answer = [self.mappings["vocab"]["word_2_wid"][word] for word in self.mappings["aid_2_answer"][answer_id].split()]
        answer = [global_vars.BOS_IDX] + answer + [global_vars.EOS_IDX]
        
        answer_vqa = answer_vqa_id
        img_path = os.path.join(self.imgs_path, map_name, "agent_view_frames", img_num + ".jpeg")
        
        img = Image.open(img_path)
        img = self.trainTransform(img)

        return (question, img), (answer, answer_vqa)

def collate_fn(batch):

    questions = []
    imgs = []
    answers = []
    answers_vqa = []
    answers_len = []

    for (ques, img), (answer, answer_vqa) in batch:
        questions.append(ques)
        imgs.append(img)
        answers_vqa.append(answer_vqa)

        answers.append(torch.LongTensor(answer))
        answers_len.append(len(answer))
    
    imgs = torch.stack(imgs, dim=0)
    answers_vqa = torch.from_numpy(np.array(answers_vqa))
    answers_len = torch.from_numpy(np.array(answers_len))

    answers = torch.nn.utils.rnn.pad_sequence(answers, batch_first=True, padding_value=global_vars.PAD_IDX)

    return (questions, imgs), (answers, answers_vqa)


        



    