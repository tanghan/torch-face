set -e
export PYTHONPATH=${WORKING_PATH}:$PYTHONPATH
date
export PYTHONUNBUFFERED=0
cd /running_package/torch-face
python3 tools/dist_cluster/url2IP.py
cat /job_data/mpi_hosts
dis_url=$(head -n +1 /job_data/mpi_hosts)
#mpirun -n 16 -ppn 8 --hostfile /job_data/mpi_hosts python3 tools/train/dist_baseline_train.py  --gpu_num 8 --rec_path /cluster_home/data/train_data/abtdge_id1w/abtdge_id1w_Above9_miabove10_20200212.rec --idx_path /cluster_home/data/train_data/abtdge_id1w/abtdge_id1w_Above9_miabove10_20200212.idx --batch_size 200 --sample_rate 1. --backbone_lr_ratio 1. --loss_type ArcFace --dist-url tcp://$dis_url:8000
#env variable ${LOCAL_OUTPU} dir can save data of you job, after exec it will be upload to hadoop_out path 
#mpirun -n 16 -ppn 8 --hostfile /job_data/mpi_hosts python3 tools/train/dist_baseline_train.py --gpu_num 8 --rec_path /cluster_home/data/train_data/abtdge_id1w/abtdge_id1w_Above9_miabove10_20200212.rec --idx_path /cluster_home/data/train_data/abtdge_id1w/abtdge_id1w_Above9_miabove10_20200212.idx --batch_size 200 --sample_rate 1. --fc_prefix /cluster_home/weight_imprint/glint360k_mega/abtdge_id1w/16_gpu --weights_path /cluster_home//pretrain_models/glint360k_mega_norm/backbone-249228.pth --backbone_lr_ratio 0.1 --resume --loss_type ArcFace --dist-url tcp://$dis_url:8000
mpirun -n 16 -ppn 8 --hostfile /job_data/mpi_hosts python3 tools/train/dist_kd_baseline_train_ds.py --gpu_num 8 --rec_path /cluster_home/data/train_data/abtdge_id1w/abtdge_id1w_Above9_miabove10_20200212.rec --idx_path /cluster_home/data/train_data/abtdge_id1w/abtdge_id1w_Above9_miabove10_20200212.idx --batch_size 200 --sample_rate 1. --weights_path /cluster_home//pretrain_models/teacher/circleloss/backbone-76520.pth --backbone_lr_ratio 1. --loss_type CircleLoss --dist-url tcp://$dis_url:8000 --fp16
#python3 tools/train/baseline_train.py --gpu_num 8 --rec_path /cluster_home/data/train_data/abtdge_id1w/abtdge_id1w_Above9_miabove10_20200212.rec --idx_path /cluster_home/data/train_data/abtdge_id1w/abtdge_id1w_Above9_miabove10_20200212.idx --batch_size 200 --sample_rate 1. --fc_prefix /cluster_home/weight_imprint/abtdge_id1w/imprint_weight --weights_path /cluster_home//pretrain_models/glint360k_norm/backbone-333800.pth --backbone_lr_ratio 1. --resume
#python3 tools/train/baseline_train.py --gpu_num 8 --rec_path /cluster_home/data/train_data/id_card/njn/small_njn.rec --idx_path /cluster_home/data/train_data/id_card/njn/small_njn.idx --batch_size 160 --sample_rate 0.1 --weights_path /cluster_home/pretrain_models/glint360k_norm/backbone-333800.pth --resume --fc_prefix /cluster_home/weight_imprint/id_card --resume --backbone_lr_ratio 0.1
#python3 tools/train/baseline_train.py --gpu_num 8 --rec_path /cluster_home/data/train_data/id_card/njn/small_njn.rec --idx_path /cluster_home/data/train_data/id_card/njn/small_njn.idx --batch_size 160 --sample_rate 0.1 --weights_prefix /cluster_home/weight_imprint/id_card
#python3 tools/train/baseline_train.py --gpu_num 8 --rec_path /cluster_home/data/train_data/id_card/njn/small_njn.rec --idx_path /cluster_home/data/train_data/id_card/njn/small_njn.idx --batch_size 160 --sample_rate 0.1
#CurricularFace
