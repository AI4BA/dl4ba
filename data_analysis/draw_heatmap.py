import numpy as np
import pandas
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker

df = pandas.read_excel('res/rq4/firefox_top10_cliff.xlsx',header=None)
data = np.array(df)

# alg = ['Bi-LSTM-A+ELMo','Bi-LSTM+ELMo','Bi-LSTM-A+BERT','Bi-LSTM+BERT','Bi-LSTM-A+Glove','Bi-LSTM+Glove','MLP+Glove','LSTM-A+BERT','LSTM+W2V','LSTM+Glove','MLP+BERT','Bi-LSTM-A+W2V','TextCNN+W2V','TextCNN+ELMo','Bi-LSTM+W2V','LSTM-A+ELMo','MLP+W2V','TextCNN+Glove','LSTM-A+Glove','LSTM+ELMo','LSTM+BERT','LSTM-A+W2V','TextCNN+BERT','MLP+ELMo']

# colors = ['#9E9E9E', '#FFFFFF']
# bounds = [0, 0.05, 1]

colors = ['#8ECFC9', '#FFBE7A', '#FA7F6F', '#82B0D2']
bounds = [0, 0.147, 0.33, 0.474, 1]
color_labels = ['negligible', 'small', 'medium', 'large']

cmap = mcolors.ListedColormap(colors)

def hua(data, bounds, cmap, color_labels):
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    plt.imshow(data, cmap=cmap, norm=norm, interpolation='nearest')
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=6)

    cbar.set_ticks([0.0735, 0.2385, 0.402, 0.737])
    cbar.ax.set_yticklabels(color_labels)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            plt.text(j, i, format(data[i, j], '.3f'),
                    horizontalalignment='center',
                    verticalalignment='center',
                    color='black',
                    fontsize=5)

    plt.tick_params(axis='both', which='both', labelsize=6)
    ax = plt.gca()
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    plt.show()

hua(data,bounds,cmap,color_labels)