python3 url2IP.py
cat /job_data/mpi_hosts
dis_url=$(head -n +1 /job_data/mpi_hosts)
mpirun -n 4 -ppn 2 --hostfile /job_data/mpi_hosts bash ${WORKING_PATH}/custom_cmds.sh --dist-url tcp://$dis_url:8000 --launcher mpi 
