for i in {1..3}
do
	python -m main @scripts/args_humanoid.txt --logdir_suffix BC --order C --updates_per_step 0 --seed $i --no_pretrain_qrisk --num_task_transitions 27000 --bc
done

for i in {1..3}
do
	python -m main @scripts/args_anymal.txt --logdir_suffix BC --order C --updates_per_step 0 --seed $i --no_pretrain_qrisk --num_task_transitions 11000 --bc
done

for i in {1..3}
do
	python -m main @scripts/args_allegro.txt --logdir_suffix BC --order C --updates_per_step 0 --seed $i --no_pretrain_qrisk --num_task_transitions 30000 --bc
done