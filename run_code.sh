cd src
        python train_and_eval.py --normtype max --diffusion_steps 10 --train_all True
	python train_and_eval.py --diffusion_steps 10 --train_all True
	
for i in $(seq 2 8);
do
	a=$((10*i))
	python train_and_eval.py --normtype max --diffusion_steps $a
	python train_and_eval.py --diffusion_steps $a
done
