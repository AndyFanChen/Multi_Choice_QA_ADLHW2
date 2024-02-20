from argparse import ArgumentParser, Namespace
from pathlib import Path
import json
from dataset import Multi_Dataset
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, BertForMultipleChoice, BertTokenizerFast

from transformers import AutoModelForMultipleChoice, AutoTokenizer

from tqdm.auto import tqdm
from tqdm import trange
from multi_test import MultiTest

# %%
# model = BertForMultipleChoice.from_pretrained("bert-base-chinese").to("cuda")
# "hfl/chinese-roberta-wwm-ext"
tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
# tokenizer = AutoTokenizer.from_pretrained("luhua/chinese_pretrain_mrc_roberta_wwm_ext_large")
# model = AutoModelForMultipleChoice.from_pretrained(model_checkpoint)
# from transformers import AutoTokenizer
# tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")


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


def oneEpochTrain(model, scheduler, optimizer, train_loader):
    device = args.device
    tok_cor = 0
    tok_n = 0
    loss_sum = 0
    loss_count = 0
    step = 0
    accumulation_steps = 2
    eval_steps = 100

    eval_loss_sum = 0
    eval_loss_count = 0
    eval_tok_cor = 0
    eval_tok_n = 0

    bar = tqdm(train_loader, desc="Train")
    for batch in bar:
        batch['input_ids'] = batch['input_ids'].to(device)
        batch['token_type_ids'] = batch['token_type_ids'].to(device)
        batch['attention_mask'] = batch['attention_mask'].to(device)
        batch['labels'] = batch['labels'].to(device)

        output = model(input_ids=batch['input_ids'], token_type_ids=batch['token_type_ids'],
                       attention_mask=batch['attention_mask'], labels=batch['labels'])
        ans_label = torch.argmax(output.logits, dim=1)

        # ============================================acc Counting
        batch_cor = (batch['labels'].eq(ans_label.view_as(batch['labels']))).sum().long().item()
        tok_cor += batch_cor
        tok_n += batch['labels'].size(0)
        # ============================================acc Counting
        # ============================================lossAndStep
        loss = output.loss
        # 是算整個batch的loss
        thisLossCount = batch['labels'].size(0)
        loss_sum += loss
        loss_count += thisLossCount
        loss.backward()
        step += 1
        # 改gradient Acc
        if step % accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        # if args.grad_clip > 0.0:
        #     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)

        # ============================================================eval
        eval_loss_sum += loss
        eval_loss_count += thisLossCount
        eval_tok_cor += batch_cor
        eval_tok_n += batch['labels'].size(0)
        if step % eval_steps == 0:
            accRate = tok_cor / (tok_n + 1E-8)
            avg_loss = loss_sum / loss_count
            print('Train Loss: {:6.4f} Acc: {:6.4f} ({}/{})'.format(avg_loss, accRate, tok_cor, tok_n))

            eval_accRate = eval_tok_cor / (eval_tok_n + 1E-8)
            eval_avg_loss = eval_loss_sum / eval_loss_count

            print(
                'This {} step Train Loss: {:6.4f} Acc: {:6.4f} ({}/{})'.format(eval_steps, eval_avg_loss, eval_accRate,
                                                                               eval_tok_cor, eval_tok_n))

            eval_loss_sum = 0
            eval_loss_count = 0
            eval_tok_cor = 0
            eval_tok_n = 0
        bar.set_postfix(lr=optimizer.param_groups[0]['lr'])

        # =========================================================eval

        # 測試用!
        # if step == 50:
        #     return

    accRate = tok_cor / (tok_n + 1E-8)
    avg_loss = loss_sum / loss_count
    print('Train Loss: {:6.4f} Acc: {:6.4f} ({}/{})'.format(avg_loss, accRate, tok_cor, tok_n))


def oneEpochValid(model, valid_loader):
    device = args.device
    tok_cor = 0
    tok_n = 0
    loss_sum = 0
    loss_count = 0
    step = 0
    eval_steps = 100
    with torch.no_grad():
        for batch in tqdm(valid_loader, desc="Valid"):
            batch['input_ids'] = batch['input_ids'].to(device)
            batch['token_type_ids'] = batch['token_type_ids'].to(device)
            batch['attention_mask'] = batch['attention_mask'].to(device)
            batch['labels'] = batch['labels'].to(device)

            output = model(input_ids=batch['input_ids'], token_type_ids=batch['token_type_ids'],
                           attention_mask=batch['attention_mask'], labels=batch['labels'])
            ans_label = torch.argmax(output.logits, dim=1)

            # ============================================acc Counting
            batch_cor = (batch['labels'].eq(ans_label.view_as(batch['labels']))).sum().long().item()
            tok_cor += batch_cor
            tok_n += batch['labels'].size(0)
            # ============================================acc Counting
            # ============================================lossAndStep
            loss = output.loss
            # 是算整個batch的loss
            thisLossCount = batch['labels'].size(0)
            loss_sum += loss
            loss_count += thisLossCount
            step += 1
            if step % eval_steps == 0:
                accRate = tok_cor / (tok_n + 1E-8)
                avg_loss = loss_sum / loss_count
                print('Valid Loss: {:6.4f} Acc: {:6.4f} ({}/{})'.format(avg_loss, accRate, tok_cor, tok_n))

            # if step == 50:
            #     return

    accRate = tok_cor / (tok_n + 1E-8)
    avg_loss = loss_sum / loss_count
    print('Valid Loss: {:6.4f} Acc: {:6.4f} ({}/{})'.format(avg_loss, accRate, tok_cor, tok_n))

    return accRate


def saveCKPT(model):
    model_save_dir = "saved_Multi_model_1109"
    model.save_pretrained(model_save_dir)
    print("Model Save")


def main(args):
    # data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    # "bert-base-chinese"
    # "hfl/chinese-roberta-wwm-ext"
    model = AutoModelForMultipleChoice.from_pretrained("hfl/chinese-roberta-wwm-ext").to(args.device)
    # model =  AutoModelForMultipleChoice.from_pretrained("bert-base-uncased").to(args.device)
    # model.init_weights()

    tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")

    train_data_paths = args.data_dir / "train.json"
    valid_data_paths = args.data_dir / "valid.json"
    context_data_paths = args.data_dir / "context.json"

    train_data = json.loads(train_data_paths.read_text(encoding='utf-8'))
    valid_data = json.loads(valid_data_paths.read_text(encoding='utf-8'))
    context_data = json.loads(context_data_paths.read_text(encoding='utf-8'))

    train_data_tokenized = tokenizeQuestion(train_data)
    valid_data_tokenized = tokenizeQuestion(valid_data)
    context_tokenized = tokenizer(context_data, add_special_tokens=False)

    train_set = Multi_Dataset(train_data, train_data_tokenized, context_tokenized)
    valid_set = Multi_Dataset(valid_data, valid_data_tokenized, context_tokenized)

    # 查pin_memory用法
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    # testData
    test_data_paths = args.data_dir / "test.json"
    test_data = json.loads(test_data_paths.read_text(encoding='utf-8'))

    # test_data_tokenized = tokenizeQuestion(test_data)
    # test_set = Multi_Dataset(test_data, valid_data_tokenized, context_tokenized)
    # test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    learning_rate = args.lr
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    # 把scheduler改成對特定目標的sche
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=400, gamma=0.95)

    # epoch_pbar = trange(args.num_epoch, desc="Epoch")
    # best_acc = 0
    for epoch in range(args.num_epoch):
        oneEpochTrain(model, scheduler, optimizer, train_loader)
        oneEpochValid(model, valid_loader)
        saveCKPT(model)

    # testProcess = MultiTest(test_data, context_data, args.batch_size, model, tokenizer, args.device)
    # result = testProcess.testResult()
    # print(result)


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
    parser.add_argument("--lr", type=float, default=1e-4)

    # data loader
    parser.add_argument("--batch_size", type=int, default=8)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    parser.add_argument("--num_epoch", type=int, default=1)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    # args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)