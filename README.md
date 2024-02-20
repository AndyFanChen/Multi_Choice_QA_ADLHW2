# 2022 ADLHW2

## Step1 download
this code will download the model from dropbox.
```
bash download.sh
```

## Step2 unzip and open the floder
After download and unzip the floder you can run this.
```
unzip r10723050ADLHW2.zip?dl=1
cd r10723050ADLHW2/src
```
## Step3 run the code
run in this format to use model trained to general answer:
```
python3.9 QA_outputAns.py --context_file "${1}" --test_file "${2}" --pred_file "${3}"
```
If using the data contained in download.sh, you can run by: 
```
python3.9 QA_outputAns.py --context_file "./Data/context.json" --test_file "./Data/test.json" --pred_file "MultAndQA_final.csv"
```

## Step4 run other model
If you want to use base model, scratch model to predict: 
base:

```
python3.9 QA_outputAns.py --context_file "./Data/context.json" --test_file "./Data/test.json" --pred_file "MultAndQA_final.csv" --Multi_model "./saved_Multi_model_1109_base" --QA_model "./saved_QA_model_1109_base"
```
scratch:
```
python3.9 QA_outputAns.py --context_file "./Data/context.json" --test_file "./Data/test.json" --pred_file "MultAndQA_final.csv" --Multi_model "./saved_Multi_model_1109_strach" --QA_model "./saved_QA_model_1109_strach"
```