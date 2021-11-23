for SEED in 1 2 3 4 5;
do python train.py --seed $SEED --task_name $task;
done;
