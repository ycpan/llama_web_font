#python scripts/openai_server_demo/openai_api_server.py --base_model /home/user/panyongcan/project/big_model/chinese-alpaca-2-7b --lora_model /home/user/panyongcan/project/Chinese-LLaMA-Alpaca-2/scripts/training/sft_output_dir/checkpoint-550/sft_lora_model --gpus 0
#python openai_api_server.py --base_model /home/user/panyongcan/project/big_model/chinese-alpaca-2-13b  --gpus 0 --load_in_4bit
#python openai_api_server.py --base_model /home/user/panyongcan/project/Chinese-LLaMA-Alpaca-2/scripts/merge_llama2_alpaca --lora_model /home/user/panyongcan/project/Chinese-LLaMA-Alpaca-2/scripts/training/sft_output_dir/sft_lora_model --gpus 0 --load_in_4bit
python openai_api_server.py --base_model merge_llama2_alpaca  --gpus 0 --load_in_4bit
#python openai_api_server.py --base_model /home/user/panyongcan/project/big_model/Llama-2-70b-instruct-v2/  --gpus 0 --load_in_8bit



