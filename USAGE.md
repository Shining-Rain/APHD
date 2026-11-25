APHD

APHD requires PyTorch, NumPy, and SciPy. For GPU training, install a CUDA-enabled build of PyTorch compatible with your NVIDIA driver.

Download the three files per dataset from the [README](README.md#datasets). Save them under `../datasets/Data/douban-book/`, `../datasets/Data/douban-movie/`, and `../datasets/Data/yelp/`, respectively. From the APHD repository root, prepare the validation splits:

    python prepare_data.py --source ../datasets/Data

Then train on Douban-Book with the command below. Use douban-movie or yelp to train on either of the other datasets.

    python main.py --dataset douban-book

Each dataset has one 7:1:2 training, validation, and test split. APHD trains on the training set and uses validation performance to select the number of epochs. It then refits a freshly initialized model on the combined training and validation sets for that many epochs and evaluates it once on the test set.
