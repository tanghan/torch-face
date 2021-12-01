set -e
export PYTHONPATH=${WORKING_PATH}:$PYTHONPATH
date
env
pip3 list
export PYTHONUNBUFFERED=0
cd ${WORKING_PATH}
python3 url2IP.py
cat /job_data/mpi_hosts
dis_url=$(head -n +1 /job_data/mpi_hosts)
mpirun -n 6 -ppn 3 --hostfile /job_data/mpi_hosts python3 my_test.py --config a.txt --dist-url tcp://$dis_url:12345 --launcher mpi 
#mpirun -n 4 -ppn 2 --hostfile /job_data/mpi_hosts python3 my_test.py --config a.txt --dist-url $dis_url --launcher mpi 
