python generation_speed.py --model_name_or_path /home/liujing/External/model/llama/llama2-7b --quantize_config_save_dir /home/liujing/External/model/llama/llama2-7b-4bit-128g

python generation_speed.py --model_name_or_path /home/liujing/External/model/llama/llama2-7b --from_pretrained

# fp
python generation_speed_batch.py --model_name_or_path /home/liujing/External/model/llama/llama2-7b --from_pretrained

# w4 only
python generation_speed_batch.py --model_name_or_path /home/liujing/External/model/llama/llama2-7b

# fp
python prefill_speed.py --model_name_or_path /home/liujing/External/model/llama/llama2-7b --from_pretrained

# w4 only
python prefill_speed.py --model_name_or_path /home/liujing/External/model/llama/llama2-7bd
