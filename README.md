
## Installing Anaconda + Python 3.11 + Tensorflow 2.15 + Keras 3 on Windows and CUDA

Based on: [Tensorflow GPU (Latest 2.14) installation on Windows 11 through WSL2](https://www.youtube.com/watch?v=VE5OiQSfPLg) with modifications to make everything work against the [compatibilitymatrix](https://www.tensorflow.org/install/source#gpu)

**General:**

The following steps are based on the compatibility matrix here: <https://www.tensorflow.org/install/source#gpu>
Impossible to follow not backwards compatible guide from Nvidia: <https://docs.nvidia.com/deeplearning/cudnn/latest/installation/linux.html>

**On windows:**

1. Install wsl2
2. On windows machine install only Nvidia Game Ready Drivers. (NO CUDA TOOLKIT)
3. Install Ubuntu 20.04 in wsl2
4. Download: Anaconda For Linux: (Anaconda serves a similar purpose to nvm for NodeJS)
<https://www.anaconda.com/download> - for other versions
<https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh>
5. Download: cuDNN for Ubuntu 20.04
<https://developer.nvidia.com/rdp/cudnn-archive> - for other versions
<https://developer.nvidia.com/downloads/compute/cudnn/secure/8.9.6/local_installers/11.x/cudnn-local-repo-ubuntu2004-8.9.6.50_1.0-1_amd64.deb/>
6\. Place the downloaded files into ~/workspace/ in your wsl2 environment for ease of access (we will be using this directory to install in wsl2)

**In wsl2:**

1.  `sudo apt-get update`
2.  `sudo apt-get upgrade`
3.  `cd ~/workspace/`
4.  `bash Anaconda3-2024.02-1-Linux-x86\_64.sh`
5. `conda config --set auto\_activate\_base false`
6.  `nvidia-smi`
7.  `conda --version`
8.  `conda create -n py31 python=3.11`
9.  `conda activate py31`
			Should return 3.11x:
10.  `python --version `

Following steps are found here: [https://developer.nvidia.com/cuda-toolkit-archive*](https://developer.nvidia.com/cuda-toolkit-archive)*

11.  `wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86\_64/cuda-wsl-ubuntu.pin`
12.  `sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600`
13.  `wget https://developer.download.nvidia.com/compute/cuda/12.2.0/local\_installers/cuda-repo-wsl-ubuntu-12-2-local\_12.2.0-1\_amd64.deb`
14.  `sudo dpkg -i cuda-repo-wsl-ubuntu-12-2-local\_12.2.0-1\_amd64.deb`
15.  `sudo cp /var/cuda-repo-wsl-ubuntu-12-2-local/cuda-\*-keyring.gpg /usr/share/keyrings/`
16.  `sudo apt-get update`
17.  `sudo apt-get -y install cuda`
18.  `cd ~/`
19.  `cd workspace/`
20.  `sudo dpkg -i cudnn-local-repo-ubuntu2004-8.9.6.50\_1.0-1\_amd64.deb`
21.  `sudo cp /var/cudnn-local-\*/cudnn-\*-keyring.gpg /usr/share/keyrings/`
22.  `sudo apt-get update`

If using other versions of(cudnn, cudatoolkit, python, ubuntu), you can ensure availability of the following packages here: <https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/> (might need to replace ubuntu version in this link if you are using other ubuntu distros)

23.  `sudo apt-get install libcudnn8=8.9.6.50-1+cuda12.2`
24.  `sudo apt-get install libcudnn8-dev=8.9.6.50-1+cuda12.2`
25.  `sudo apt-get install libcudnn8-samples=8.9.6.50-1+cuda12.2`
26.  `sudo reboot`
27.  `cd ~/`
28.  `conda activate py31`
29.  `python3 -m pip install tensorflow[and-cuda]`
30.  `pip install --ignore-installed --upgrade tensorflow==2.15`

*If tensorflow actually sees the GPU after some output you should see this line: [PhysicalDevice(name='/physical\_device:GPU:0', device\_type='GPU')]:*

31.  `python3 -c "import tensorflow as tf; print(tf.config.list\_physical\_devices('GPU'))"`

## Installing in Docker for Windows and CUDA

We have to install Docker into WSL**

1.  Remove **Docker Desktop** from Windows
2. Make sure `docker ps` does not work in WSL
3.  reboot WSL: close all terminals and VSCode that uses WSL
4. Run in **Windows PowerShell** `wsl --shutdown` and wait **8** seconds - let it close
5. See wsl instances disappear in SysInternals Process Explorer
6. Start WSL terminal - it will launch WSL automatically

[Install Docker Ubuntu guide](https://docs.docker.com/engine/install/ubuntu/)

Don't install docker-compose as a separate package. Use `docker compose` command with space, which is a newer way.

[Install NVidia Container Toolkit](https://gist.github.com/atinfinity/f9568aa9564371f573138712070f5bad)

NOTE: Skip **Install Docker CLI** section, since you already installed docker correctly

Now you can pass gpus into docker container
Test by running `sudo docker run --gpus all nvcr.io/nvidia/k8s/cuda-sample:nbody nbody -gpu -benchmark`

Note the `--gpus all` flag. It works only after you install **nvidia-container-toolkit**

## Learning materials

**In order of attack:**

**Math:**

- Derivatives
[Introduction to Derivatives](https://www.mathsisfun.com/calculus/derivatives-introduction.html)
- Python:
- Numpy crash course
[Python NumPy Tutorial for Beginners](https://www.youtube.com/watch?v=QUT1VHiLmmI)**

**Machine Learning:**

- General machine learning concepts
[Machine Learning Tutorial Python - 2: Linear Regression Single Variable](https://www.youtube.com/watch?v=8jazNUpO3lQ&list=PLeo1K3hjS3uvCeTYTeyfe0-rN5r8zn9rw)

**Deep Learning:**

- Easy to follow general explanation of NNs
[Neural Networks / Deep Learning - YouTube](https://www.youtube.com/playlist?list=PLblh5JKOoLUIxGDQs4LFFD--41Vzf-ME1)
- Introduction to Deep Learning by MIT, (duplicate lectures for different years)
[MIT Introduction to Deep Learning | 6.S191](https://www.youtube.com/watch?v=ErnWZxJovaM&list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI)
- Introduction to deep learning with Tensorflow 2 and Keras
[Deep Learning With Tensorflow 2.0, Keras and Python - YouTube](https://www.youtube.com/playlist?list=PLeo1K3hjS3uu7CxAacxVndI4bE_o3BDtO)


**Reinforcement Learning:**

- Bellman Equation
- Dynamic Programming
- Monte Carlo Reinforcement
- Q Learning
- Deep Q Learning
- Policy Gradient Methods
- Actor Critic (A3C)
- Continuous Action Space Actor Critic
- Proximal Policy Optimization (PPO)
- [Bellman Equation Basics for Reinforcement Learning](https://www.youtube.com/watch?v=14BfO5lMiuk&list=PLWzQK00nc192L7UMJyTmLXaHa3KcO0wBT)
- **Markov Decision Processes
- Bellman Equation
- Q Learning
- Deep Q Learning
- Double  Deep Q Learning
[Intro to RL](https://www.youtube.com/watch?v=cVTud58UfpQ&list=PLYgyoWurxA_8ePNUuTLDtMvzyf-YW7im2)

**Full course on Reinforcement learning (University of Waterloo, 2018):**

**Hard to follow, sometimes too much theory, explanations are shallow (for us) in a lot of cases, but a very good course to get a general understanding of all theoretical aspects of reinforcement learning.**
[https://www.youtube.com/watch?v=yOWBb0mqENw&list=PLdAoL1zKcqTXFJniO3Tqqn6xMBBL07EDc
](https://www.youtube.com/watch?v=yOWBb0mqENw&list=PLdAoL1zKcqTXFJniO3Tqqn6xMBBL07EDc)

**Github:**

**Rainbow DQN, Tensorflow 2.0, incomplete implementation, crappy performance:**
<https://github.com/Huixxi/TensorFlow2.0-for-Deep-Reinforcement-Learning>
**Rainbow DQN, Pytorch uses convolutional layers:**
<https://github.com/Kaixhin/Rainbow>
**Rainbow DQN**, **Tensorflow 2.0:**
<https://github.com/ClementPerroud/Rainbow-Agent>

**Papers:**

- **DoubleQ-learning** : Adding a Target Network that is used in the loss function and upgrade once every tau steps. See paper[ ](https://arxiv.org/abs/1509.06461)[Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)
- **Distributional RL** : Approximating the probability distributions of the Q-values instead of the Q-values themself. See paper :[ ](https://arxiv.org/abs/1707.06887)[A Distributional Perspective on Reinforcement Learning](https://arxiv.org/abs/1707.06887)
- **Prioritizedreplay** : Sampling method that prioritize experiences with big *Temporal Difference(TD) errors* (~loss) at the beginning of a training. See paper :[ ](https://arxiv.org/abs/1511.05952)[Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)
- **Dueling Networks**: Divide neural net stream into two branches, an action stream and a value stream. Both of them combined formed the Q-action values. See paper :[ ](https://arxiv.org/abs/1509.06461)[Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1509.06461)
- **Multi-step learning** : Making Temporal Difference bigger than classic DQN (where TD = 1). See paper[ ](https://arxiv.org/abs/1703.01327)[Multi-step Reinforcement Learning: A Unifying Algorithm](https://arxiv.org/abs/1703.01327)
- **NoisyNets** : Replace classic epsilon-greedy exploration/exploitation with noise in the Neural Net.[ ](https://arxiv.org/abs/1706.10295)[Noisy Networks for Exploration](https://arxiv.org/abs/1706.10295)
- **Rainbow DQN:** <https://arxiv.org/pdf/1710.02298>
- **Categories of Reinforcement Learning Algorithms:** <https://arxiv.org/pdf/2209.14940>
