from argparse import ArgumentParser, Namespace
from pathlib import Path
import json
from dataset import QA_Dataset
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, BertForMultipleChoice, BertTokenizerFast
from tqdm.auto import tqdm
from tqdm import trange
import numpy

class QATest:

    def __init__(self, test_data, context_data, relList, model, tokenizer, device):
        self.test_data = test_data
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        test_ques_tokenized = self.tokenizeQuestion(test_data)
        context_tokenized = self.tokenizer(context_data, add_special_tokens=False)
        test_set = QA_Dataset("test", test_data, relList, test_ques_tokenized, context_tokenized, tokenizer)
        self.test_loader = DataLoader(test_set, batch_size=1, shuffle=False, pin_memory=True)


    def tokenizeQuestion(self, data):
        questions = []
        for data_ele in data:
            questions.append(data_ele["question"])
        # questions = [[train_data_ele["question"] * 4 for train_data_ele in train_data]
        questions_tokenized = self.tokenizer(questions, add_special_tokens=False)
        return questions_tokenized

    def evaluate(self, batch, output):
        # There is a bug and room for improvement in postprocessing
        # Hint: Open your prediction file to see what is wrong

        answer = ''
        max_prob = float('-inf')
        num_of_windows = batch['input_ids'].shape[0]
        # doc_stride = batch['doc_stride']
        # paragraph = batch['paragraph']

        for k in range(num_of_windows):
            # Obtain answer by choosing the most probable start position / end position
            start_prob, start_index = torch.max(output.start_logits[k], dim=0)
            end_prob, end_index = torch.max(output.end_logits[k], dim=0)
            # token_type_id = batch['token_type_ids'][k].detach().numpy()
            # paragraph_start = token_type_id.argmax()
            # paragraph_end = len(token_type_id) - 1 - token_type_id[::-1].argmax()-1
            # Probability of answer is calculated as sum of start_prob and end_prob
            prob = start_prob + end_prob

            # Replace answer if calculated probability is larger than previous windows
            if prob > max_prob:
                if start_index > end_index:
                    continue
                else:
                    max_prob = prob
                # origin_start = start_index + k * doc_stride - paragraph_start
                # origin_end = end_index + k * doc_stride - paragraph_start
                    # Convert tokens to chars (e.g. [1920, 7032] --> "大 金")
                    answer = self.tokenizer.decode(batch['input_ids'][k][start_index: end_index + 1])
        # if '[UNK]' in answer:
        #     print('发现 [UNK]，这表明有文字无法编码, 使用原始文本')
        #     #print("Paragraph:", paragraph)
        #     #print("Paragraph:", paragraph_tokenized.tokens)
        #     print('--直接解码预测:', answer)
        #     #找到原始文本中对应的位置
        #     raw_start =  batch['input_ids'][k].token_to_chars(origin_start)[0]
        #     raw_end = batch['input_ids'][k].token_to_chars(origin_end)[1]
        #     answer = paragraph[raw_start:raw_end]
        #     print('--原始文本预测:',answer)
        # Remove spaces in answer (e.g. "大 金" --> "大金")
        # print("answer")
        # print(answer)
        return answer.replace(' ', '')



    def testResult(self):
        # 做test_list


        answer_list = []
        for batch in tqdm(self.test_loader, desc="Test"):


            batch['input_ids'] = batch['input_ids'].squeeze(dim=0)
            batch['token_type_ids'] = batch['token_type_ids'].squeeze(dim=0)
            batch['attention_mask'] = batch['attention_mask'].squeeze(dim=0)

            batch['input_ids'] = batch['input_ids'].to(self.device)
            batch['token_type_ids'] = batch['token_type_ids'].to(self.device)
            batch['attention_mask'] = batch['attention_mask'].to(self.device)
            # batch['start'] = batch['start'].to(self.device)
            # batch['end'] = batch['end'].to(self.device)

            with torch.no_grad():
                output_dict = self.model(input_ids=batch['input_ids'], token_type_ids=batch['token_type_ids'],
                attention_mask=batch['attention_mask'])
            answer = self.evaluate(batch, output_dict)
            answer_list.append(answer)

        return answer_list


