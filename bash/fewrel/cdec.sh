export PYTHONPATH=`pwd`

epoch1=8
epoch2=8

CUDA_VISIBLE_DEVICES=0 python3 main/cdec.py \
  --memory_size 20 \
  --total_round 5 \
  --task_name FewRel \
  --data_file data/data_with_marker.json \
  --relation_file data/id2rel.json \
  --num_of_train 420 \
  --num_of_val 140 \
  --num_of_test 140 \
  --batch_size_per_step 32 \
  --exp_name CDec_s1_epoch${epoch1}_s2_epoch${epoch2} \
  --num_of_relation 80  \
  --cache_file data/fewrel_data.pt \
  --rel_per_task 8 \
  --step1_epochs ${epoch1} \
  --step2_epochs ${epoch2}
