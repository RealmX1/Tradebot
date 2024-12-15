create new conda environment with `conda create -n trading_env python=3.12`
open the environment with `conda activate trading_env`
run `pip install -r requirements.txt` to install the dependencies.

run `python prepare_data.py` to prepare the data.
run `python train.py` to train the model.
run `python test.py` to test the model on unseen data.