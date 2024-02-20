from argparse import ArgumentParser, Namespace
from pathlib import Path
import json
from dataset import Multi_Dataset
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, BertForMultipleChoice, BertTokenizerFast
from tqdm.auto import tqdm
from tqdm import trange


class MultiTest:

    def __init__(self, test_data, context_data, batch_size, model, tokenizer, device):
        self.test_data = test_data
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        test_data_tokenized = self.tokenizeQuestion(test_data)
        context_tokenized = self.tokenizer(context_data, add_special_tokens=False)
        test_set = Multi_Dataset(test_data, test_data_tokenized, context_tokenized)
        self.test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=True)


    def tokenizeQuestion(self, data):
        questions = []
        for data_ele in data:
            questions.append(data_ele["question"])
        # questions = [[train_data_ele["question"] * 4 for train_data_ele in train_data]
        questions_tokenized = self.tokenizer(questions, add_special_tokens=False)
        return questions_tokenized

    def testResult(self):
        # 做test_list
        result_list = []
        for batch in tqdm(self.test_loader, desc="Test"):
            # print('input_ids')
            # print(batch['input_ids'].shape)
            # print('labels')
            # print(batch['labels'].shape)
            batch['input_ids'] = batch['input_ids'].to(self.device)
            batch['token_type_ids'] = batch['token_type_ids'].to(self.device)
            batch['attention_mask'] = batch['attention_mask'].to(self.device)
            batch['labels'] = batch['labels'].to(self.device)
            with torch.no_grad():
                output_dict = self.model(input_ids=batch['input_ids'], token_type_ids=batch['token_type_ids'],
                attention_mask=batch['attention_mask'], labels=batch['labels'])
            ans_label = torch.argmax(output_dict.logits, dim=1)
            result_list += ans_label

        return result_list


