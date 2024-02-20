from argparse import ArgumentParser, Namespace
from pathlib import Path
import json
from dataset import QA_Dataset
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, BertForMultipleChoice, BertForQuestionAnswering, BertTokenizerFast
from tqdm.auto import tqdm
from tqdm import trange
from multi_test import MultiTest
from QA_test import QATest
import csv
from transformers import AutoTokenizer, AutoModelForMultipleChoice, AutoModelForQuestionAnswering
import numpy

from transformers import AutoConfig, AutoModel




# %%
# tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")
# tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
# hfl/chinese-roberta-wwm-ext
tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext-large")
# tokenizer = AutoTokenizer.from_pretrained("luhua/chinese_pretrain_mrc_roberta_wwm_ext_large")
# tokenizer = AutoTokenizer.from_pretrained("luhua/chinese_pretrain_mrc_macbert_large")
# %%

"""
讀資料
讀出question的文字跟選項的文字
把文字tokenized
把tokenized前後的東西丟入Dataset

每次Dataloader抓一個batch->其中的每個item->
__getitem__:
    讀tokenized_question跟答案
    讀答案的位置(編號)
    拼接question跟候選答案(放好101, 102後丟入padding function中)
    把padding好的東西轉torch後與正確label一起輸出
"""


def tokenizeQuestion(data):
    questions = []
    for data_ele in data:
        questions.append(data_ele["question"])
    # questions = [[train_data_ele["question"] * 4 for train_data_ele in train_data]
    questions_tokenized = tokenizer(questions, add_special_tokens=False)
    return questions_tokenized



def oneEpochTrain(model, optimizer, scheduler, train_loader):
    device = args.device
    tok_cor = 0
    tok_n = 0
    loss_sum = 0
    loss_count = 0
    step = 0
    accumulation_steps = 8
    eval_steps = 50

    eval_loss_sum = 0
    eval_loss_count = 0
    eval_tok_cor = 0
    eval_tok_n = 0
    totalLossList = []
    difLossList = []
    totalEMList = []
    difEMList = []

    bar = tqdm(train_loader, desc="Train")
    for batch in bar:
        
        batch['input_ids'] = batch['input_ids'].to(device)
        batch['token_type_ids'] = batch['token_type_ids'].to(device)
        batch['attention_mask'] = batch['attention_mask'].to(device)
        batch['start'] = batch['start'].to(device)
        batch['end'] = batch['end'].to(device)


        output = model(input_ids=batch['input_ids'], token_type_ids=batch['token_type_ids'],
                       attention_mask=batch['attention_mask'], start_positions=batch['start'], end_positions=batch['end'])

        start_index = torch.argmax(output.start_logits, dim=1)
        end_index = torch.argmax(output.end_logits, dim=1)


        # ============================================acc Counting
        batch_cor = ((start_index == batch['start']) & (end_index == batch['end'])).sum().long().item()
        tok_cor += batch_cor
        tok_n += batch['start'].size(0)
        # ============================================acc Counting
        # ============================================lossAndStep
        loss = output.loss

        thisLossCount = batch['start'].size(0)
        loss_sum += loss
        loss_count += thisLossCount
        loss.backward()
        step += 1
        # 改gradient Acc
        if step % accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        eval_loss_sum += loss
        eval_loss_count += thisLossCount
        eval_tok_cor += batch_cor
        eval_tok_n += batch['input_ids'].size(0)
        
        
        if step % eval_steps == 0:
            accRate = tok_cor / (tok_n + 1E-8)
            avg_loss = loss_sum / loss_count
            totalEMList.append(accRate)
            totalLossList.append(avg_loss)

            print('Train Loss: {:6.4f} Acc: {:6.4f} ({}/{})'.format(avg_loss, accRate, tok_cor, tok_n))

            
            eval_accRate = eval_tok_cor / (eval_tok_n + 1E-8)
            eval_avg_loss = eval_loss_sum / eval_loss_count

            difEMList.append(eval_accRate)
            difLossList.append(eval_avg_loss)
            print('This {} step Train Loss: {:6.4f} Acc: {:6.4f} ({}/{})'.format(eval_steps, eval_avg_loss, eval_accRate, eval_tok_cor, eval_tok_n))

            eval_loss_sum = 0
            eval_loss_count = 0
            eval_tok_cor = 0
            eval_tok_n = 0

        bar.set_postfix(lr=optimizer.param_groups[0]['lr'])

        # 測試用!
        # if step == 300:
        #     return totalLossList, difLossList, totalEMList, difEMList

    accRate = tok_cor / (tok_n + 1E-8)
    avg_loss = loss_sum / loss_count
    print('Train Loss: {:6.4f} Acc: {:6.4f} ({}/{})'.format(avg_loss, accRate, tok_cor, tok_n))
    print(totalLossList, difLossList)
    return totalLossList, difLossList, totalEMList, difEMList


def evaluate(batch, output):

    answer = ''
    max_prob = float('-inf')
    num_of_windows = batch['input_ids'].shape[0]

    for k in range(num_of_windows):
        # Obtain answer by choosing the most probable start position / end position
        start_prob, start_index = torch.max(output.start_logits[k], dim=0)
        end_prob, end_index = torch.max(output.end_logits[k], dim=0)

        prob = start_prob + end_prob

        # Replace answer if calculated probability is larger than previous windows
        # 可改對答案的長度要合理控制
        if prob > max_prob:
            if start_index > end_index:
                continue
            else:
                max_prob = prob

            answer = tokenizer.decode(batch['input_ids'][k][start_index: end_index + 1])

                
    return answer.replace(' ', '')


def oneEpochValid(model, valid_loader):
    device = args.device
    tok_cor = 0
    tok_n = 0
    loss_sum = 0
    loss_count = 0
    step = 0
    eval_steps = 50

    # batch size只有1
    with torch.no_grad():
        for batch in tqdm(valid_loader, desc="Valid"):

            batch['input_ids'] = batch['input_ids'].squeeze(dim=0)
            batch['token_type_ids'] = batch['token_type_ids'].squeeze(dim=0)
            batch['attention_mask'] = batch['attention_mask'].squeeze(dim=0)
            # batch['start'] = batch['start']
            # batch['end'] = batch['end']

            batch['input_ids'] = batch['input_ids'].to(device)
            batch['token_type_ids'] = batch['token_type_ids'].to(device)
            batch['attention_mask'] = batch['attention_mask'].to(device)
            # batch['start'] = batch['start'].to(device)
            # batch['end'] = batch['end'].to(device)
            # print(batch['input_ids'].shape)
            # 可以不給ans去跑model!
            output = model(input_ids=batch['input_ids'], token_type_ids=batch['token_type_ids'],
                        attention_mask=batch['attention_mask'])

            pred_answer = evaluate(batch, output)


            # ============================================acc Counting
            batch_cor = pred_answer == batch["answer"][0]
            # if pred_answer == batch["answer"][0]:
            #     print("correct!")
            #     print(batch_cor)
            tok_cor += batch_cor
            # print(tok_cor)
            tok_n += 1
            # print(tok_n)
            # ============================================acc Counting
            # ============================================lossAndStep

            step += 1
            if step % eval_steps == 0:
                accRate = tok_cor / (tok_n + 1E-8)
                # avg_loss = loss_sum / loss_count
                print('Valid Acc: {:6.4f} ({}/{})'.format(accRate, tok_cor, tok_n))
            

            # if step == 300:
            #     return

    accRate = tok_cor / (tok_n + 1E-8)
    # avg_loss = loss_sum / loss_count
    print('Valid Acc: {:6.4f} ({}/{})'.format(accRate, tok_cor, tok_n))

    return accRate

def saveCKPT(model):
    # model_save_dir = "saved_QA_model_Train"
    model_save_dir = "saved_QA_model_1109_no_pre_weight"
    model.save_pretrained(model_save_dir)
    print("Model Save")


def main(args):
    # data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    
    model = AutoModelForQuestionAnswering.from_pretrained("hfl/chinese-roberta-wwm-ext-large").to(args.device)
    # config = AutoConfig.from_pretrained('bert-base-uncased')
    # config.hidden_size = 64
    # config.num_hidden_size = 4
    # config.num_hidden_layer = 2
    # model =  AutoModelForQuestionAnswering.from_pretrained("bert-base-uncased").to(args.device)
    # model.init_weights()
    # model = AutoModelForQuestionAnswering.from_pretrained("hfl/chinese-roberta-wwm-ext-large").to(args.device)
    # model = AutoModelForQuestionAnswering.from_pretrained("luhua/chinese_pretrain_mrc_macbert_large").to(args.device)
    # tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext-large")
    # tokenizer = AutoTokenizer.from_pretrained("luhua/chinese_pretrain_mrc_macbert_large")

    train_data_paths = args.data_dir / "train.json"
    valid_data_paths = args.data_dir / "valid.json"
    context_data_paths = args.data_dir / "context.json"

    train_data = json.loads(train_data_paths.read_text(encoding='utf-8'))
    valid_data = json.loads(valid_data_paths.read_text(encoding='utf-8'))
    context_data = json.loads(context_data_paths.read_text(encoding='utf-8'))

    train_data_tokenized = tokenizeQuestion(train_data)
    valid_data_tokenized = tokenizeQuestion(valid_data)
    context_tokenized = tokenizer(context_data, add_special_tokens=False)

    train_set = QA_Dataset("train", train_data, 0, train_data_tokenized, context_tokenized, tokenizer)
    valid_set = QA_Dataset("valid", valid_data, 0, valid_data_tokenized, context_tokenized, tokenizer)

    # 查pin_memory用法
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False, pin_memory=True)

    # testData
    test_data_paths = args.data_dir / "test.json"
    test_data = json.loads(test_data_paths.read_text(encoding='utf-8'))

    # test_data_tokenized = tokenizeQuestion(test_data)
    # test_set = Multi_Dataset(test_data, valid_data_tokenized, context_tokenized)
    # test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    
    learning_rate = args.lr
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)

    # epoch_pbar = trange(args.num_epoch, desc="Epoch")
    # best_acc = 0

    # for epoch in range(1):
    for epoch in range(args.num_epoch):
        totalLossList, difLossList, totalEMList, difEMList = oneEpochTrain(model, optimizer, scheduler, train_loader)
        oneEpochValid(model, valid_loader)

        saveCKPT(model)
        print(totalLossList)
        for i in range(len(totalLossList)):
            totalLossList[i] = totalLossList[i].item()
        for i in range(len(difLossList)):
            difLossList[i] = difLossList[i].item()
        print(totalLossList)
        for i in range(len(totalEMList)):
            totalEMList[i] = totalEMList[i]
        for i in range(len(difEMList)):
            difEMList[i] = difEMList[i]
        with open("totalLoss{}.csv".format(epoch), 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(['totalLoss'])
            writer.writerow(totalLossList)

        with open("difLoss{}.csv".format(epoch), 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(['difLoss'])
            writer.writerow(difLossList)
        
        with open("totalEM{}.csv".format(epoch), 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(['totalEM'])
            writer.writerow(totalEMList)

        with open("difEM{}.csv".format(epoch), 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(['difEM'])
            writer.writerow(difEMList)

    # 讀Multi model後出List結果，再把list結果套入
    multiModel = AutoModelForMultipleChoice.from_pretrained("./saved_Multi_model_1109_strach").to(args.device)

    # # multiModel = multiModel.load
    print("loading_multi_test")
    multTestProcess = MultiTest(test_data, context_data, 1, multiModel, tokenizer, args.device)
    relList = multTestProcess.testResult()
    print("predicting_QA_test")
    QATestProcess = QATest(test_data, context_data, relList, model, tokenizer, args.device)
    answerList = QATestProcess.testResult()

    # ========================================存答案!
    all_ids = []
    for allQues in test_data:
        all_ids.append(allQues['id'])
    with open(args.pred_file, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['id','answer'])
        writer.writerows(zip(all_ids, answerList))


    # 紀錄answerList
    # print(answerList)


"""
test檔logic:
建立data
建立model
讀model
建立接結果的list
for batch in testLoader:
做預測 -> 存結果到list
"""
    # print("Saving Model ...")
    # model_save_dir = "saved_model"
    # model.save_pretrained(model_save_dir)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./Data/",
    )
    # parser.add_argument(
    #     "--cache_dir",
    #     type=Path,
    #     help="Directory to the preprocessed caches.",
    #     default="./cache/slot/",
    # )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/",
    )

    # # loss
    # parser.add_argument('--grad_clip', default = 5., type=float, help='max gradient norm')

    # optimizer
    parser.add_argument("--lr", type=float, default=3e-5)

    # data loader
    parser.add_argument("--batch_size", type=int, default=16)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    parser.add_argument("--num_epoch", type=int, default=4)

    parser.add_argument("--pred_file", type=Path, default="MultAndQA_1109_no_pre_weight.csv")
 

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    # args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)