python run_infer.py \
--gpu='1' \
--nr_types=6 \
--type_info_path=type_info.json \
--batch_size=64 \
--model_mode=fast \
--model_path=../pretrained/hovernet_fast_pannuke_type_tf2pytorch.tar \
--nr_inference_workers=8 \
--nr_post_proc_workers=16 \
tile \
--input_dir=dataset/sample_tiles/imgs/ \
--output_dir=dataset/sample_tiles/pred/ \
--mem_usage=0.1 \
--draw_dot \
--save_qupath

python run_infer.py --gpu='0,1' --nr_types=5 --model_mode=fast --model_path=hovernet_fast_monusac_type_tf2pytorch.tar --type_info_path=type_info.json
tile --input_dir=/data1/wyj/M/datasets/MoNuSAC/masks-new/ --output_dir=./outputtest/
--mem_usage=0.1
--draw_dot
--save_qupath

--gpu=0,1 --nr_types=5 --batch_size=2 --model_mode=fast --model_path=hovernet_fast_monusac_type_tf2pytorch.tar --type_info_path=type_info.json tile --input_dir=/data1/wyj/M/datasets/MoNuSACCROP/images/ --output_dir=./outputtest/ --mem_usage=0.1 --draw_dot --save_qupath
python run_infer.py --gpu=0,1 --nr_types=5 --model_mode=fast --model_path=hovernet_fast_monusac_type_tf2pytorch.tar --type_info_path=type_info.json
tile --input_dir=/data1/wyj/M/datasets/MoNuSAC/masks-new/ --output_dir=./outputtest/
--mem_usage=0.1
--draw_dot
--save_qupath
--gpu=0,1 --nr_types=3 --batch_size=2 --model_mode=fast --model_path=/data1/wyj/M/samples/PRM/hover_net-master/logs/01/net_epoch=50.tar --type_info_path=type_info.json tile
--input_dir=/data1/wyj/M/datasets/MoNuSACCROP/images/ --output_dir=./outputtest/ --mem_usage=0.1 --draw_dot --save_qupath --save_raw_map
--gpu=0,1 --view=train