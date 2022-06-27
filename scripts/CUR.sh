for i in {1..3}
do
	python -m main @scripts/args_humanoid.txt --logdir_suffix CUR --seed $i --warmup_penalty 1000
done
for i in {1..3}
do
	python -m main @scripts/args_anymal.txt --logdir_suffix CUR --seed $i --warmup_penalty 250
done
for i in {1..3}
do
	python -m main @scripts/args_allegro.txt --logdir_suffix CUR --seed $i --warmup_penalty 2500
done

