ALGO=rlnmc #options: sa/nmc/rlnmc

n_devices=4 #adapt to your devices
n_replicas=2048 #adapt to your devices

problem_batches=(0) #hyperparameter optimization/training instances
# problem_batches=(1 2 3 4 5) #benchmarking instances

for ((a=0; a<${#problem_batches[@]}; a++))
do
	python3.11 bench.py --bench_config config_bench_sf250 --problem_config config_250 --id ${problem_batches[a]} --algo $ALGO --save_name run_$ALGO --n_devices $n_devices --n_replicas $n_replicas
done