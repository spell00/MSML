
## Authors

* Simon Pelletier (2021-) (Current Maintainer)

# Prerequisites on ubuntu
`apt-get install -y parallel`<br/>
`apt-get install -y python3`<br/>
`apt-get install -y python3-pip`<br/>
`apt-get install -y r-base`<br/>
`apt-get purge -y openjdk-\*`<br/>
`apt install -y openjdk-8-jre`<br/>
`apt install -y openjdk-8-jdk`<br/>
`apt-get install -y ant`<br/>
`apt-get install -y ca-certificates-java`<br/>
`update-ca-certificates -f`<br/>
`chmod +x mzdb2train.sh`<br/>

`chmod +x msml/scripts/mzdb2tsv/amm`

# Install python dependencies
`pip install -r requirements.txt`


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

`python3 msml\dl\train\mlp\train_ae_classifier.py --triplet_loss=1 --predict_tests=1 --dann_sets=0 --balanced_rec_loader=0 --dann_plates=0 --zinb=0 --variational=0 --use_valid=1 --use_test=1`

For your data to work, it should be a matrix: rows are samples, columns are features. Feature names can be whatever,
but the row names (in the first column named ID), the names should be as such: `{experiment_name}_{class}_{batch_number}_{id}`

*** The batch number should start with the letter `p`, followed by batch number. This is because for the experiment
it was designed for, the batches were the plates in which the bacteria grew. It should change soon!
e.g.: `rd159_blk_p16_09`

## Observe results from a server on a local machine 
On local machine:<br/>
`ssh -L 16006:127.0.0.1:6006 simonp@192.168.3.33`

On server:<br/>
`python3 -m tensorboard.main --logdir=/path/to/log/file`

Open in browser:<br/>
`http://127.0.0.1:16006/`
