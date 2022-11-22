
#generate 5000 iteration images on sr attack
# CUDA_VISIBLE_DEVICES=$GPU python data_generate.py attack=invertgradients_sr attack.optim.max_iterations=5000 case.model="resnet18" attack.init="zeros" num_trials=302 attack.save.out_dir='/home/zx/Gitrepo/breaching/rec_datasets/scale_4'


#generate 5000 iteration images on normal attack
# CUDA_VISIBLE_DEVICES=$GPU python data_generate.py attack=invertinggradients attack.optim.max_iterations=5000 case.model="resnet18" attack.init="zeros" num_trials=302 attack.save.out_dir='/home/zx/Gitrepo/breaching/rec_datasets/normal'  


# - attack=invertgradients_sr
# - attack.optim.max_iterations=1000
# - case.model=resnet18
# - attack.init=zeros
# - num_trials=1000
# - case.user.user_idx=396



#generate more data
# max_iter=5000

# starts=(2..10)

# (for start in {1..2}
# do
#     GPU=3
#     mkdir "/home/zx/Gitrepo/breaching/rec_datasets/normal/$start" 

#     mkdir "/home/zx/Gitrepo/breaching/rec_datasets/normal/$start/hq" 
#     mkdir "/home/zx/Gitrepo/breaching/rec_datasets/normal/$start/lq_$max_iter" 

#     CUDA_VISIBLE_DEVICES=$GPU python data_generate.py attack=invertinggradients attack.optim.max_iterations=$max_iter case.model="resnet18" attack.init="zeros" num_trials=1000 case.user.data_points_start=$start attack.save.out_dir="/home/zx/Gitrepo/breaching/rec_datasets/normal/$start"  
# done)&


# (for start in {3..4}
# do
#     GPU=4
#     mkdir "/home/zx/Gitrepo/breaching/rec_datasets/normal/$start" 

#     mkdir "/home/zx/Gitrepo/breaching/rec_datasets/normal/$start/hq" 
#     mkdir "/home/zx/Gitrepo/breaching/rec_datasets/normal/$start/lq_$max_iter" 

#     CUDA_VISIBLE_DEVICES=$GPU python data_generate.py attack=invertinggradients attack.optim.max_iterations=$max_iter case.model="resnet18" attack.init="zeros" num_trials=1000 case.user.data_points_start=$start attack.save.out_dir="/home/zx/Gitrepo/breaching/rec_datasets/normal/$start"  
# done)&



# (for start in {5..6}
# do
#     GPU=5
#     mkdir "/home/zx/Gitrepo/breaching/rec_datasets/normal/$start" 

#     mkdir "/home/zx/Gitrepo/breaching/rec_datasets/normal/$start/hq" 
#     mkdir "/home/zx/Gitrepo/breaching/rec_datasets/normal/$start/lq_$max_iter" 

#     CUDA_VISIBLE_DEVICES=$GPU python data_generate.py attack=invertinggradients attack.optim.max_iterations=$max_iter case.model="resnet18" attack.init="zeros" num_trials=1000 case.user.data_points_start=$start attack.save.out_dir="/home/zx/Gitrepo/breaching/rec_datasets/normal/$start"  
# done)&

# (for start in {7..9}
# do
#     GPU=7
#     mkdir "/home/zx/Gitrepo/breaching/rec_datasets/normal/$start" 

#     mkdir "/home/zx/Gitrepo/breaching/rec_datasets/normal/$start/hq" 
#     mkdir "/home/zx/Gitrepo/breaching/rec_datasets/normal/$start/lq_$max_iter" 

#     CUDA_VISIBLE_DEVICES=$GPU python data_generate.py attack=invertinggradients attack.optim.max_iterations=$max_iter case.model="resnet18" attack.init="zeros" num_trials=1000 case.user.data_points_start=$start attack.save.out_dir="/home/zx/Gitrepo/breaching/rec_datasets/normal/$start"  
# done)&


# generate more data 1000
# max_iter=1000
# for start in {1..8}
# do
#     GPU=`expr "$start" - "1"`
#     mkdir "/root/data/GitRepo/breaching/rec_datasets/scale_4/$start" 

#     mkdir "/root/data/GitRepo/breaching/rec_datasets/scale_4/$start/hq" 
#     mkdir "/root/data/GitRepo/breaching/rec_datasets/scale_4/$start/lq_$max_iter" 

#     CUDA_VISIBLE_DEVICES=$GPU python data_generate.py attack=invertgradients_sr attack.optim.max_iterations=$max_iter case.model="resnet18" attack.init="zeros" num_trials=1000 case.user.data_points_start=$start attack.save.out_dir="/root/data/GitRepo/breaching/rec_datasets/scale_4/$start" &
#     sleep 5
#     # echo $GPU
# done

# echo $max_iter >> log 


# CUDA_VISIBLE_DEVICES=0 python data_generate.py attack=invertgradients_sr attack.optim.max_iterations=$max_iter case.model="resnet18" attack.init="zeros" num_trials=1000 case.user.data_points_start=1 attack.save.out_dir="/root/data/GitRepo/breaching/rec_datasets/scale_4/1" &
# sleep 2

# CUDA_VISIBLE_DEVICES=1 python data_generate.py attack=invertgradients_sr attack.optim.max_iterations=1000 case.model="resnet18" attack.init="zeros" num_trials=1000 case.user.data_points_start=2 attack.save.out_dir="/root/data/GitRepo/breaching/rec_datasets/scale_4/2"
# sleep 2

# CUDA_VISIBLE_DEVICES=2 python data_generate.py attack=invertgradients_sr attack.optim.max_iterations=1000 case.model="resnet18" attack.init="zeros" num_trials=1000 case.user.data_points_start=3 attack.save.out_dir="/root/data/GitRepo/breaching/rec_datasets/scale_4/3" 
# sleep 2

# CUDA_VISIBLE_DEVICES=3 python data_generate.py attack=invertgradients_sr attack.optim.max_iterations=1000 case.model="resnet18" attack.init="zeros" num_trials=1000 case.user.data_points_start=4 attack.save.out_dir="/root/data/GitRepo/breaching/rec_datasets/scale_4/4" 
# sleep 2

# CUDA_VISIBLE_DEVICES=4 python data_generate.py attack=invertgradients_sr attack.optim.max_iterations=1000 case.model="resnet18" attack.init="zeros" num_trials=1000 case.user.data_points_start=5 attack.save.out_dir="/root/data/GitRepo/breaching/rec_datasets/scale_4/5" 
# sleep 2

# CUDA_VISIBLE_DEVICES=5 python data_generate.py attack=invertgradients_sr attack.optim.max_iterations=1000 case.model="resnet18" attack.init="zeros" num_trials=1000 case.user.data_points_start=6 attack.save.out_dir="/root/data/GitRepo/breaching/rec_datasets/scale_4/6" 
# sleep 2

# CUDA_VISIBLE_DEVICES=6 python data_generate.py attack=invertgradients_sr attack.optim.max_iterations=1000 case.model="resnet18" attack.init="zeros" num_trials=1000 case.user.data_points_start=7 attack.save.out_dir="/root/data/GitRepo/breaching/rec_datasets/scale_4/7" 
# sleep 2

# CUDA_VISIBLE_DEVICES=7 python data_generate.py attack=invertgradients_sr attack.optim.max_iterations=1000 case.model="resnet18" attack.init="zeros" num_trials=1000 case.user.data_points_start=8 attack.save.out_dir="/root/data/GitRepo/breaching/rec_datasets/scale_4/8" 
# sleep 2

# max_iter=1000
# for start in {5..8}
# do
#     GPU=`expr "$start" - "1"`
#     mkdir "/root/data/GitRepo/breaching/rec_datasets/test/$start" 

#     mkdir "/root/data/GitRepo/breaching/rec_datasets/test/$start/hq" 
#     mkdir "/root/data/GitRepo/breaching/rec_datasets/test/$start/lq_$max_iter" 

#     CUDA_VISIBLE_DEVICES=$GPU python data_generate.py attack=invertgradients_sr attack.optim.max_iterations=$max_iter case.model="resnet18" attack.init="zeros" num_trials=1000 case.user.data_points_start=$start attack.save.out_dir="/root/data/GitRepo/breaching/rec_datasets/test/$start" & 
#     sleep 5
#     # echo $GPU
# done



# CUDA_VISIBLE_DEVICES=0 python data_generate.py attack=invertgradients_sr attack.optim.max_iterations=1000 case.model="resnet18" attack.init="zeros" num_trials=1 case.user.data_points_start=1 attack.save.out_dir="/root/data/GitRepo/breaching/rec_datasets/test/1"
# 




#----------------------------------------------------Test diffusion and grad combination-------------------------------
#

# CUDA_VISIBLE_DEVICES=5 python optim_diffusion_debug.py --class_cond False --max_iterations 10 --timestep_respacing "50"
# CUDA_VISIBLE_DEVICES=5 python optim_diffusion_debug.py --class_cond False --max_iterations 10 --timestep_respacing "150"
# CUDA_VISIBLE_DEVICES=5 python optim_diffusion_debug.py --class_cond False --max_iterations 10 --timestep_respacing "250"
# CUDA_VISIBLE_DEVICES=5 python optim_diffusion_debug.py --class_cond False --max_iterations 50 --timestep_respacing "50"
# CUDA_VISIBLE_DEVICES=5 python optim_diffusion_debug.py --class_cond False --max_iterations 50 --timestep_respacing "150"
# CUDA_VISIBLE_DEVICES=5 python optim_diffusion_debug.py --class_cond False --max_iterations 50 --timestep_respacing "250"
# CUDA_VISIBLE_DEVICES=5 python optim_diffusion_debug.py --class_cond False --max_iterations 100 --timestep_respacing "50"
# CUDA_VISIBLE_DEVICES=5 python optim_diffusion_debug.py --class_cond False --max_iterations 100 --timestep_respacing "150"
# CUDA_VISIBLE_DEVICES=5 python optim_diffusion_debug.py --class_cond False --max_iterations 100 --timestep_respacing "250"


# CUDA_VISIBLE_DEVICES=5 python optim_diffusion_debug.py --class_cond True --max_iterations 10 --timestep_respacing "50"
# CUDA_VISIBLE_DEVICES=5 python optim_diffusion_debug.py --class_cond True --max_iterations 10 --timestep_respacing "150"
# CUDA_VISIBLE_DEVICES=5 python optim_diffusion_debug.py --class_cond True --max_iterations 10 --timestep_respacing "250"
# CUDA_VISIBLE_DEVICES=5 python optim_diffusion_debug.py --class_cond True --max_iterations 50 --timestep_respacing "50"
# CUDA_VISIBLE_DEVICES=5 python optim_diffusion_debug.py --class_cond True --max_iterations 50 --timestep_respacing "150"
# CUDA_VISIBLE_DEVICES=5 python optim_diffusion_debug.py --class_cond True --max_iterations 50 --timestep_respacing "250"
# CUDA_VISIBLE_DEVICES=5 python optim_diffusion_debug.py --class_cond True --max_iterations 100 --timestep_respacing "50"
# CUDA_VISIBLE_DEVICES=5 python optim_diffusion_debug.py --class_cond True --max_iterations 100 --timestep_respacing "150"
# CUDA_VISIBLE_DEVICES=5 python optim_diffusion_debug.py --class_cond True --max_iterations 100 --timestep_respacing "250"



#benchmark
#

# samle_flag="attack.diffusion.class_cond=False, \
#     attack.diffusion.model_path=/home/zx/data/GitRepo/breaching/breaching/attacks/accelarate/guided_diffusion/models/256x256_diffusion.pt, \
#     attack.diffusion.timestep_respacing=50, \
#     "

# sample_flag="attack.diffusion.class_cond=False attack.diffusion.timestep_respacing='50'"

# grad_attack_flag="attack.optim.max_iterations=10"

# reconstruction_flag="attack.save.out_dir=/home/zx/data/GitRepo/breaching/out/diffusion/benchmark case.data.partition=unique-class case.model=resnet18 case.data.examples_from_split=validation "


# CUDA_VISIBLE_DEVICES=2 python data_generate_diffusion.py attack=invertinggradients_diffusion $sample_flag $grad_attack_flag $reconstruction_flag
# # CUDA_VISIBLE_DEVICES=0 python data_generate_diffusion.py attack=invertinggradients_diffusion $sample_flag $grad_attack_flag $reconstruction_flag dryrun=True