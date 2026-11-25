APHD

APHD requires PyTorch, NumPy, and SciPy. For GPU training, install a CUDA-enabled build of PyTorch compatible with your NVIDIA driver.

Download the datasets linked in the [README](README.md#datasets). From the APHD repository root, prepare the validation splits before training (replace `../datasets/Data` with your downloaded data directory):

    python prepare_data.py --source ../datasets/Data

Then train on Douban-Book with the command below. Use douban-movie or yelp to train on either of the other datasets.

    python main.py --dataset douban-book

Each dataset has one 7:1:2 training, validation, and test split. APHD trains on the training set and uses validation performance to select the number of epochs. It then refits a freshly initialized model on the combined training and validation sets for that many epochs and evaluates it once on the test set.
