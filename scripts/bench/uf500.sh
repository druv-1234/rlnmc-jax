ALGO=rlnmc #options: sa/nmc/rlnmc

n_devices=4 #adapt to your devices
n_replicas=1024 #adapt to your devices

problem_batches=(0) #hyperparameter optimization/training instances
# problem_batches=(1 2 3 4 5) #benchmarking instances

for ((a=0; a<${#problem_batches[@]}; a++))
do
	python bench.py --bench_config config_bench_uf500 --problem_config config_500 --id ${problem_batches[a]} --algo $ALGO --save_name run_$ALGO --n_devices $n_devices --n_replicas $n_replicas
done