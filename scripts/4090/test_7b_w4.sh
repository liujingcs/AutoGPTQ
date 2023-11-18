for n in 1 2 3 4 5
do
SAVE_PATH=output/llama-7b/w4only
mkdir -p ${SAVE_PATH}
python prefill_speed.py --model_name_or_path /data/dongzhiwei/models/llama2-13b 2>&1 | tee -a ${SAVE_PATH}/train_r${r}_n${n}.log
done