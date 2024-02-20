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


def main(args):
    test_data_paths = args.test_file
    test_data = json.loads(test_data_paths.read_text(encoding='utf-8'))
    context_data_paths = args.context_file
    context_data = json.loads(context_data_paths.read_text(encoding='utf-8'))
    # context_tokenized = tokenizer(context_data, add_special_tokens=False)
    # tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    tokenizer = AutoTokenizer.from_pretrained("./saved_tokenizer_large")

    multiModel = AutoModelForMultipleChoice.from_pretrained(args.Multi_model).to(args.device)
    qaModel = AutoModelForQuestionAnswering.from_pretrained(args.QA_model).to(args.device)

    # multiModel = multiModel.load
    print("loading_multi_test")
    multTestProcess = MultiTest(test_data, context_data, args.batch_size, multiModel, tokenizer, args.device)
    relList = multTestProcess.testResult()
    print("predicting_QA_test")
    QATestProcess = QATest(test_data, context_data, relList, qaModel, tokenizer, args.device)
    answerList = QATestProcess.testResult()

    # ========================================存答案!
    # write prediction to file (args.pred_file)
    # if args.pred_file.parent:
    #     args.pred_file.parent.mkdir(parents=True, exist_ok=True)

    all_ids = []
    for allQues in test_data:
        all_ids.append(allQues['id'])
    with open(args.pred_file, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['id', 'answer'])
        writer.writerows(zip(all_ids, answerList))
        # for id, answer in zip(all_ids, answerList):
        #     f.write("{},{}\n".format(id, answer))


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

    parser.add_argument(
        "--context_file",
        type=Path,
        help="Directory to the dataset.",
        default="./Data/context.json",
    )

    parser.add_argument(
        "--test_file",
        type=Path,
        help="Directory to the dataset.",
        default="./Data/test.json",
    )

    parser.add_argument(
        "--Multi_model",
        type=Path,
        help="Directory to the dataset.",
        default="./saved_Multi_model_1109",
    )

    parser.add_argument(
        "--QA_model",
        type=Path,
        help="Directory to the dataset.",
        default="./saved_QA_model_1109_loss",
    )

    # optimizer
    parser.add_argument("--lr", type=float, default=4e-5)

    # data loader
    parser.add_argument("--batch_size", type=int, default=1)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    parser.add_argument("--num_epoch", type=int, default=1)

    parser.add_argument("--pred_file", type=Path, default="MultAndQA_final.csv")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    # args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)