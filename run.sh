# cd r10723050ADLHW2/src
# python3.9 QA_outputAns.py --context_file "./Data/context.json" --test_file "./Data/test.json" --pred_file "MultAndQA_final.csv"
python3.9 QA_outputAns.py --context_file "${1}" --test_file "${2}" --pred_file "${3}"



# model:
#  --Multi_model "./saved_Multi_model_1109" --QA_model "./saved_QA_model_1109_loss"

# base:
# python3.9 QA_outputAns.py --context_file "./Data/context.json" --test_file "./Data/test.json" --pred_file "MultAndQA_final.csv" --Multi_model "./saved_Multi_model_1109_base" --QA_model "./saved_QA_model_1109_base"

# scratch:
# python3.9 QA_outputAns.py --context_file "./Data/context.json" --test_file "./Data/test.json" --pred_file "MultAndQA_final.csv" --Multi_model "./saved_Multi_model_1109_strach" --QA_model "./saved_QA_model_1109_strach"
