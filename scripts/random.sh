for i in {1..3}
do
	python -m main @scripts/args_humanoid.txt --logdir_suffix random --allocation random --seed $i --std_dev 0.05 --action_budget 20000
done
for i in {1..3}
do
	python -m main @scripts/args_allegro.txt --logdir_suffix random --allocation random --seed $i --action_budget 19000
done
for i in {1..3}
do
	python -m main @scripts/args_anymal.txt --logdir_suffix random --allocation random --seed $i --std_dev 0.05 --action_budget 4000
done
