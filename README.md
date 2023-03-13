# Mario_RL

# setup
Creating the environment:
```console
conda create --prefix /nfs/scratch_2/koen/mario_rl python=3.9
```

Activate environment:
```console
conda activate /nfs/scratch_2/koen/mario_rl
```

Install pytorch:
```console
conda install pytorch==1.13.0 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia
```

Install requirements
```console
pip install -r requirements.txt
```

Check whether Mario Bros works by running
```console
gym_super_mario_bros -m human
```