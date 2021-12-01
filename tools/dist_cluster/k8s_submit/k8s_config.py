""" Job Settings """
job_name = "torch-hdlt-k8s-example"
job_password = "newk8s666"

num_machines = 1
num_gpus_per_machine = 4

framework = "pytorch"
task_label = "HDLT"
project_id = "AM2018-R72"

priority = 5
docker_image = (
    "docker.hobot.cc/imagesys/torch1.7.0:" +
    "runtime-py3.6-cuda92-cudnn7.3-torch1.7.0-vision0.8.1"
)
max_jobtime = 10000  # default 7200 = 5days

# launcher only for multi-machines
launcher = "mpi"

# upload folder
upload_folder_name = "k8s_job"
folder_list = [
    "../../hdlt",
    "../../tools",
    "../../configs",
]
job_list = [
    "python3 tools/train.py --config configs/classification/resnet18.py",
]
