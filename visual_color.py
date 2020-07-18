
import matplotlib.pyplot as plt
import numpy as np
import pickle
training_file = "./traffic-signs-data/train.p"
testing_file = "./traffic-signs-data/test.p"
def plot_color_statistic(image_dataset_file=training_file, name='Dataset', sampling=10):
    with open(image_dataset_file, mode='rb') as f:
        dataset = pickle.load(f)['features']
        n = dataset.shape[0]
        np.random.shuffle(dataset)
        if n < sampling:
            sampling = n
        flattened = []
        for image in dataset[:sampling]:
            flattened.extend(image.flatten())

        std = (np.std(flattened))
        mean = (np.mean(flattened))
        num_bins = 50

        # the histogram of the data
        with plt.style.context(('seaborn-muted')):
            n, bins, patches = plt.hist(flattened, num_bins, normed=1, alpha=0.5)
            plt.xlabel('Color values')
            plt.title(r'Histogram of {} accross {} samples: $\mu={}$, $\sigma={}$'.format(name, sampling ,mean, std))

            # Tweak spacing to prevent clipping of ylabel
            plt.subplots_adjust(left=0.15)
        plt.show()