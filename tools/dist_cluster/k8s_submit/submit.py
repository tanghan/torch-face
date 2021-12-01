# Copyright (c) Horizon Robotics. All rights reserved.
# submit jobs

import argparse
import os
import subprocess

#from hdlt.common import Config
import yaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
    )
    parser.add_argument("--upload-folder", type=str, default="./")
    parser.add_argument(
        "--save-upload-folder",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        '--current-cluster',
        default=None,
        type=str,
        help='[traincli option] If None, use default in yaml'
    )
    parser.add_argument(
        '--debug', dest='debug', action='store_true', help='[traincli option]'
    )
    parser.add_argument(
        '--sleep',
        dest='sleep',
        action='store_true',
        help='sleep in cluster for debug'
    )
    return parser.parse_args()


def generate_job_yaml(cfg, upload_folder):
    yaml_name = "%s.yaml" % cfg.job_name
    with open(yaml_name, "w") as fn:
        fn.write('REQUIRED:\n')
        fn.write('  JOB_NAME: "%s"\n' % cfg.job_name)
        fn.write('  JOB_PASSWD: "%s"\n' % cfg.job_password)
        fn.write('  UPLOAD_DIR: "%s"\n' % os.path.basename(upload_folder))
        fn.write('  PROJECT_ID: "%s"\n' % cfg.project_id)
        fn.write('  WORKER_MIN_NUM: %d\n' % cfg.num_machines)
        fn.write('  WORKER_MAX_NUM: %d\n' % cfg.num_machines)
        fn.write('  GPU_PER_WORKER: %d\n' % cfg.num_gpus_per_machine)
        fn.write('  RUN_SCRIPTS: "${WORKING_PATH}/job.sh"\n')
        fn.write('OPTIONAL:\n')
        fn.write('  PRIORITY: %s\n' % cfg.priority)
        fn.write('  DOCKER_IMAGE: "%s"\n' % cfg.docker_image)
        fn.write('  WALL_TIME: %d\n' % cfg.max_jobtime)
        # set bucket
        if hasattr(cfg, 'input_bucket'):
            fn.write('  DATA_SPACE:\n')
            fn.write('    DATA_TYPE: "dmp"\n')
            fn.write('    INPUT: "%s"\n' % cfg.input_bucket)
            if hasattr(cfg, 'output_bucket') and cfg.output_bucket:
                fn.write('    OUTPUT: "%s"\n' % cfg.output_bucket)
    return yaml_name


def generate_upload_folder(cfg, upload_folder):
    assert "folder_list" in cfg
    for path in cfg.folder_list:
        if not os.path.exists(path):
            print(f"{path} not exists, skip")
            continue
        subprocess.check_call(['cp', '-r', path, upload_folder])
        print("copy %s to %s" % (path, upload_folder))


def generate_bash_file(upload_folder, run_in_sleep, job_list, launcher, num_machines, num_gpus_per_machine):
    bash_file = os.path.join(upload_folder, "job.sh")
    print("bash file: {}".format(bash_file))
    with open(bash_file, 'w') as fn:
        fn.write('set -e\n')
        fn.write('export PYTHONPATH=${WORKING_PATH}:$PYTHONPATH\n')
        fn.write('date\n')
        fn.write('env\n')
        fn.write('pip3 list\n')
        fn.write('export PYTHONUNBUFFERED=0\n')
        fn.write('cd ${WORKING_PATH}\n')
        if run_in_sleep:
            fn.write('sleep 1000000m\n')

        if num_machines > 1:  # multi-machines
            subprocess.check_call(['cp', 'url2IP.py', upload_folder])
            fn.write('python3 url2IP.py\n')
            fn.write('cat /job_data/mpi_hosts\n')
            fn.write('dis_url=$(head -n +1 /job_data/mpi_hosts)\n')
            if launcher == "mpi":
                for job in job_list:
                    command = "mpirun -n %d -ppn %d --hostfile %s %s %s" % (
                        num_machines * num_gpus_per_machine,
                        num_gpus_per_machine, "/job_data/mpi_hosts", job,
                        "--dist-url tcp://$dis_url:8000 --launcher mpi "
                    )
                    fn.write("%s\n" % command)
        else:
            for job in job_list:
                fn.write("%s\n" % job)
    subprocess.check_call(['chmod', '777', bash_file])


job_list = [
    "python3 my_test.py",
]
num_machines = 2
num_gpus_per_machine = 2

if __name__ == "__main__":
    args = parse_args()
    upload_folder_name = "k8s_job"

    args.upload_folder = os.path.join(
        args.upload_folder, upload_folder_name
    )
    subprocess.check_call(['mkdir', '-p', upload_folder_name])

    #yaml_name = generate_job_yaml(cfg, upload_folder)
    #assert os.path.exists(yaml_name), "Cannot generate yaml successfully."

    #generate_upload_folder(cfg, args.upload_folder)

    generate_bash_file(args.upload_folder, args.sleep, job_list, launcher="mpi", num_machines=num_machines, num_gpus_per_machine=num_gpus_per_machine)

    '''
    cmd = ['traincli', 'submit', '-f', yaml_name]
    if args.current_cluster:
        cmd += ['--current-cluster', args.current_cluster]
    if args.debug:
        cmd += ['--debug']
    '''

    '''
    try:
        subprocess.check_call(cmd)
        print("Submit job successfully.")
    except Exception as e:
        print(e)
        raise Exception("error")
    finally:
        if not args.save_upload_folder:
            if os.path.exists(args.upload_folder):
                subprocess.check_call(['rm', '-rf', args.upload_folder])
    '''
