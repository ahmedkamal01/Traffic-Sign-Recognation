import matplotlib.pyplot as plt
import numpy as np

def plot_instance_counts(dataset, name="dataset"):
    from collections import Counter
    counts = Counter(dataset)
    labels, values = zip(*counts.items())
    indexes = np.arange(len(labels))
    width = 0.5
    with plt.style.context(('seaborn-muted')):
        figure = plt.figure(figsize=(15, 3))
        plt.bar(indexes, values, width)
        plt.xticks(indexes + width * 0.5, labels)
        plt.xlabel('Class Label')
        plt.title('{} : Number of instance per class'.format(name))

    plt.show()