#Installing Anaconda + Python 3.11 + Tensorflow 2.15 + Keras 3
Based on: Tenserflow GPU (Latest 2.14) installation on Windows 11 through WSL2 with modifications to make everything work against the compatibilitymatrix

General:

The following steps are based on the compatibility matrix here: https://www.tensorflow.org/install/source#gpu
Impossible to follow not backwards compatible guide from Nvidia: https://docs.nvidia.com/deeplearning/cudnn/latest/installation/linux.html

On windows:

1. Install wsl2
2. On windows machine install only Nvidia Game Ready Drivers. (NO CUDA TOOLKIT)
3. Install Ubuntu 20.04 in wsl2
4. Download: Anaconda For Linux: (Anaconda serves a similar purpose to nvm for NodeJS)
	https://www.anaconda.com/download - for other versions
	https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh
5. Download: cuDNN for Ubuntu 20.04
	https://developer.nvidia.com/rdp/cudnn-archive - for other versions
https://developer.nvidia.com/downloads/compute/cudnn/secure/8.9.6/local_installers/11.x/cudnn-local-repo-ubuntu2004-8.9.6.50_1.0-1_amd64.deb/
6. Place the downloaded files into ~/workspace/ in your wsl2 environment for ease of access (we will be using this directory to install in wsl2)

In wsl2:

1  sudo apt-get update
2  sudo apt-get upgrade
3  cd ~/workspace/
4  bash Anaconda3-2024.02-1-Linux-x86_64.sh
5  conda config --set auto_activate_base false
6  nvidia-smi
7  conda --version
8  conda create -n py31 python=3.11
9  conda activate py31

	Should return 3.11x:
10  python --version 
        
	Following steps are found here: https://developer.nvidia.com/cuda-toolkit-archive
11  wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
12  sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
13  wget https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda-repo-wsl-ubuntu-12-2-local_12.2.0-1_amd64.deb
14  sudo dpkg -i cuda-repo-wsl-ubuntu-12-2-local_12.2.0-1_amd64.deb
15  sudo cp /var/cuda-repo-wsl-ubuntu-12-2-local/cuda-*-keyring.gpg /usr/share/keyrings/
16  sudo apt-get update
17  sudo apt-get -y install cuda
18  cd ~/
19  cd workspace/
20  sudo dpkg -i cudnn-local-repo-ubuntu2004-8.9.6.50_1.0-1_amd64.deb
21  sudo cp /var/cudnn-local-*/cudnn-*-keyring.gpg /usr/share/keyrings/
22  sudo apt-get update

	If using other versions of(cudnn, cudatoolkit, python, ubuntu), you can ensure availability of the following packages here: https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ (might need to replace ubuntu version in this link if you are using other ubuntu distros)
23  sudo apt-get install libcudnn8=8.9.6.50-1+cuda12.2
24  sudo apt-get install libcudnn8-dev=8.9.6.50-1+cuda12.2
25  sudo apt-get install libcudnn8-samples=8.9.6.50-1+cuda12.2
26  sudo reboot
27  cd ~/
28  conda activate py31
29  python3 -m pip install tensorflow[and-cuda]
30  pip install --ignore-installed --upgrade tensorflow==2.15

	If tensorflow actually sees the GPU after some output you should see this line: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]:
31  python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"


Nvidia in Docker
We have to install Docker into WSL
Remove Docker Desktop from Windows
Make sure `docker ps` does not work in WSL

To reboot WSL: close all terminals and VSCode that uses WSL
  Run in Windows PowerShell `wsl --shutdown` and wait 8 seconds - let it close
  See wsl instances disappear in SysInternals Process Explorer
  Start WSL terminal - it will launch WSL automatically

Install Docker Ubuntu guide
    Don't install docker-compose as a separate package. Use `docker compose` command with space, which is a newer way.
Install NVidia Container Toolkit
NOTE: Skip Install Docker CLI section, since you already installed docker correctly
Now you can pass gpus into docker container
Test by running `sudo docker run --gpus all nvcr.io/nvidia/k8s/cuda-sample:nbody nbody -gpu -benchmark`
Note the `--gpus all` flag. It works only after you install nvidia-container-toolkit



In order of attack:

Math:

- Derivatives
Introduction to Derivatives

Python:

- Numpy crash course
Python NumPy Tutorial for Beginners

Machine Learning:

- General machine learning concepts
Machine Learning Tutorial Python - 2: Linear Regression Single Variable

Deep Learning:

- Easy to follow general explanation of NNs
Neural Networks / Deep Learning - YouTube

- Introduction to Deep Learning by MIT, (duplicate lectures for different years)
MIT Introduction to Deep Learning | 6.S191

- Introduction to deep learning with Tensorflow 2 and Keras
Deep Learning With Tensorflow 2.0, Keras and Python - YouTube

Reinforcement Learning:

- Bellman Equation
- Dynamic Programming
- Monte Carlo Reinforcement
- Q Learning
- Deep Q Learning
- Policy Gradient Methods
- Actor Critic (A3C)
- Continuous Action Space Actor Critic
- Proximal Policy Optimization (PPO)
Bellman Equation Basics for Reinforcement Learning

- Markov Decision Processes
- Bellman Equation
- Q Learning
- Deep Q Learning
- Double  Deep Q Learning
Intro to RL

Full course on Reinforcement learning (University of Waterloo, 2018):
Hard to follow, sometimes too much theory, explanations are shallow (for us) in a lot of cases, but a very good course to get a general understanding of all theoretical aspects of reinforcement learning.
https://www.youtube.com/watch?v=yOWBb0mqENw&list=PLdAoL1zKcqTXFJniO3Tqqn6xMBBL07EDc


Github:

Rainbow DQN, Tensorflow 2.0, incomplete implementation, crappy performance:
https://github.com/Huixxi/TensorFlow2.0-for-Deep-Reinforcement-Learning
Rainbow DQN, Pytorch uses convolutional layers:
https://github.com/Kaixhin/Rainbow
Rainbow DQN, Tensorflow 2.0:
https://github.com/ClementPerroud/Rainbow-Agent

Papers:



DoubleQ-learning : Adding a Target Network that is used in the loss function and upgrade once every tau steps. See paper Deep Reinforcement Learning with Double Q-learning
Distributional RL : Approximating the probability distributions of the Q-values instead of the Q-values themself. See paper : A Distributional Perspective on Reinforcement Learning
Prioritizedreplay : Sampling method that prioritize experiences with big Temporal Difference(TD) errors (~loss) at the beginning of a training. See paper : Prioritized Experience Replay
Dueling Networks: Divide neural net stream into two branches, an action stream and a value stream. Both of them combined formed the Q-action values. See paper : Dueling Network Architectures for Deep Reinforcement Learning
Multi-step learning : Making Temporal Difference bigger than classic DQN (where TD = 1). See paper Multi-step Reinforcement Learning: A Unifying Algorithm
NoisyNets : Replace classic epsilon-greedy exploration/exploitation with noise in the Neural Net. Noisy Networks for Exploration
Rainbow DQN: https://arxiv.org/pdf/1710.02298
Categories of Reinforcement Learning Algorithms: https://arxiv.org/pdf/2209.14940















Categories of RL algorithms (state space / action space):





sudo apt-get install libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf

