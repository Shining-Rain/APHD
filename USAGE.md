APHD

APHD requires PyTorch, NumPy, and SciPy. For GPU training, install a CUDA-enabled build of PyTorch compatible with your NVIDIA driver.

Before the first run, extract Data.zip into the project directory so that Data/ is next to main.py. From that directory, run the following command to train on Douban-Book. Use douban-movie or yelp to train on either of the other datasets.

    python main.py --dataset douban-book

Each dataset has one 7:1:2 training, validation, and test split. APHD trains on the training set and uses validation performance to select the number of epochs. It then refits a freshly initialized model on the combined training and validation sets for that many epochs and evaluates it once on the test set.
