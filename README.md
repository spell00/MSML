
## Authors

* Simon Pelletier (2021-) (Current Maintainer)

# Prerequisites on ubuntu
apt-get install -y parallel
apt-get install -y python3
apt-get install -y python3-pip
apt-get install -y r-base
apt-get purge -y openjdk-\*
apt install -y openjdk-8-jre
apt install -y openjdk-8-jdk
apt-get install -y ant
apt-get install -y ca-certificates-java
update-ca-certificates -f
chmod +x mzdb2train.sh

chmod +x msml/scripts/mzdb2tsv/amm

# Install python dependencies
pip install -r requirements.txt


On Windows:
The first step needs to be executed on Windows because it calls raw2mzdb.exe and the software only exists for Windows.

In a  Windows PowerShell:

`./msml/preprocess/raw2mzdb.bat`

The resulting mzdb files are stored in `../../resources/mzdb/$spd/$group/`

On Linux (tested with WLS Ubuntu 20.04):

`bash ./msml/preprocess/mzdb2tsv.sh $mz_bin $rt_bin $spd $group`

The resulting tsv files are stored in `../../resources/mzdb/$spd/$group/`

## Train deep learning model
Command line example:

`python3 msml\dl\train\mlp\train_ae_classifier3.py --triplet_loss=1 --predict_tests=1`

With the default settings, the data needs to be in `'../../resources//20220706_Data_ML02/Data_FS/matrices/mz0.2/rt20.0/200spd/combat0/shift0/none/loginloop/mutual_info_classif/eco,sag,efa,kpn,blk,pool//train_inputs.csv'`

## Observe results from a server on a local machine 
On local machine:
ssh -L 16006:127.0.0.1:6006 simonp@192.168.3.33

on server: 
python3 -m tensorboard.main --logdir=/path/to/log/file

open in browser:
http://127.0.0.1:16006/
