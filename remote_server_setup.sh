#!bash

#mkdir ml
# cd ml
# git clone https://github.com/qunash/cir-ru-t5
# cd cir-ru-t5
yes | sudo apt install python3-pip
pip install transformers
pip install datasets
pip install --upgrade pip
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install flax
yes | sudo apt install nvidia-cuda-toolkit

sudo apt install ubuntu-drivers-common
sudo ubuntu-drivers autoinstall

pip install tensorboard
pip install tensorflow

#install git-lfs to be able to push to hf hub
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
git lfs install

git config --global user.email "anzoria@gmail.com"
git config --global user.name "qunash"

pip3 install torch --extra-index-url https://download.pytorch.org/whl/cu116
pip install sentencepiece
pip install sacrebleu
pip install accelerate # for auto_find_batch_size to work: https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments.auto_find_batch_size

sudo reboot
