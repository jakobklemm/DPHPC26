one time setup:

from repository root: 
python3 -m venv .venv
source .venv/bin/activate
pip install -r ml/requirements.txt

for each new terminal just do:
source .venv/bin/activate

to look at the data as a dataframe:
python -m ml.load path/to/folder/with/jakobs/zip_files

to train:
python -m ml.main train --data path/to/folder/with/jakobs/zip_files

to predict (best build for a machine, given its system.json; needs a trained model):
python -m ml.main predict --app llama.cpp --system path/to/system.json

to suggest (builds worth benchmarking next on a machine: untested, promising or uncertain):
python -m ml.main suggest --app llama.cpp --system path/to/system.json --data path/to/folder/with/jakobs/zip_files

to test: (does not make sense yet since we only have data from one machine)
python -m ml.main evaluate --data path/to/folder/with/jakobs/zip_files