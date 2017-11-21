wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.44-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1604_8.0.44-1_amd64.deb
sudo apt-get update
sudo apt-get install cuda-8-0
# then set environment variables; ideally add these in ~/.bashrc
echo "PATH and LD_LIBRARY_PATH set temporarily. Remember to add them to ~/.bashrc."
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda-8.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}} 
export PATH=/usr/local/cuda/lib64:/usr/local/cuda-8.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}} 
