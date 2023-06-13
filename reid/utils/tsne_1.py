from sklearn.manifold import TSNE
# import pandas as pd
import matplotlib.pyplot as mp

import matplotlib.pyplot as plt
import scipy.io
import torch
import numpy as np
#import time
import os
import torch.nn.functional as F

result = scipy.io.loadmat('/home/wentao/project/M3L/logs/1Conv3BN/dukemtmc+msmt17v1+cuhk03--TO--market1501/test/keshihua.mat')

feature = torch.FloatTensor(result['feature'])
# featre = F.normalize(feature, dim=1)
print(feature.size())
label = result['label'][0]

# label1 = result['label_hc'][0]
# label2 = result['label_uics'][0]
# label3 = result['label_sota'][0]


for i in range(100):
    tsne = TSNE(n_components=2,learning_rate=i)
    X_tsne = tsne.fit_transform(feature)

    plt.figure(figsize=(12,6))

    # ax=plt.subplot(2, 2, 1)
    # ax.set_title("Baseline")
    a_1 = label == 0
    plt.scatter(X_tsne[a_1, 0], X_tsne[a_1, 1], c='blue')
    a_2 = label == 1
    plt.scatter(X_tsne[a_2, 0], X_tsne[a_2, 1], c='green')
    a_3 = label == 2
    plt.scatter(X_tsne[a_3, 0], X_tsne[a_3, 1], c='yellow')
    # plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=label)
    plt.xticks([])
    plt.yticks([])

    plt.savefig('/home/wentao/project/M3L/logs/1Conv3BN/dukemtmc+msmt17v1+cuhk03--TO--market1501/test/%d.jpg' % i)

    # ax1 = plt.subplot(2, 2, 2)
    # ax1.set_title("Baseline $w/$ HC")
    # a1_1 = label1 == 54
    # plt.scatter(X_tsne[a1_1, 0], X_tsne[a1_1, 1], c='blue')
    # a1_2 = label1 == 663
    # plt.scatter(X_tsne[a1_2, 0], X_tsne[a1_2, 1], c='green')
    # a1_3 = label1 == 664
    # plt.scatter(X_tsne[a1_3, 0], X_tsne[a1_3, 1], c='orange')
    # a1_4 = label1 == 665
    # plt.scatter(X_tsne[a1_4, 0], X_tsne[a1_4, 1], c='yellow')
    # a1_5 = label1 == 666
    # plt.scatter(X_tsne[a1_5, 0], X_tsne[a1_5, 1], c='deeppink')
    # a1_6 = label1 == 776
    # plt.scatter(X_tsne[a1_6, 0], X_tsne[a1_6, 1], c='greenyellow')
    # a1_7 = label1 == 777
    # plt.scatter(X_tsne[a1_7, 0], X_tsne[a1_7, 1], c='cyan')
    # a1_8 = label1 == 778
    # plt.scatter(X_tsne[a1_8, 0], X_tsne[a1_8, 1], c='fuchsia')
    # a1_9 = label1 == 779
    # plt.scatter(X_tsne[a1_9, 0], X_tsne[a1_9, 1], c='pink')
    # a1_10 = label1 == -1
    # plt.scatter(X_tsne[a1_10, 0], X_tsne[a1_10, 1], c='#808080')
    #
    # # plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=label1)
    # plt.xticks([])
    # plt.yticks([])
    #
    #
    # ax2 = plt.subplot(2, 2, 3)
    # ax2.set_title("Baseline $w/$ UCIS")
    # a2_1 = label2 == 54
    # plt.scatter(X_tsne[a2_1, 0], X_tsne[a2_1, 1], c='blue')
    # a2_2 = label2 == 82
    # plt.scatter(X_tsne[a2_2, 0], X_tsne[a2_2, 1], c='#FF4500')
    #
    # a2_3 = label2 == -1
    # plt.scatter(X_tsne[a2_3, 0], X_tsne[a2_3, 1], c='#808080')
    #
    # # plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=label2)
    # plt.xticks([])
    # plt.yticks([])


    # ax3 = plt.subplot(2, 2, 4)
    # ax3.set_title("UCF")
    # a3_1 = label3 == 54
    # plt.scatter(X_tsne[a3_1, 0], X_tsne[a3_1, 1], c='blue')
    # a3_2 = label3 == 663
    # plt.scatter(X_tsne[a3_2, 0], X_tsne[a3_2, 1], c='green')
    # a3_3 = label3 == 664
    # plt.scatter(X_tsne[a3_3, 0], X_tsne[a3_3, 1], c='orange')
    # a3_4 = label3 == 665
    # plt.scatter(X_tsne[a3_4, 0], X_tsne[a3_4, 1], c='yellow')
    # a3_5 = label3 == 666
    # plt.scatter(X_tsne[a3_5, 0], X_tsne[a3_5, 1], c='deeppink')
    # a3_6 = label3 == 776
    # plt.scatter(X_tsne[a3_6, 0], X_tsne[a3_6, 1], c='greenyellow')
    # a3_7 = label3 == 777
    # plt.scatter(X_tsne[a3_7, 0], X_tsne[a3_7, 1], c='cyan')
    # a3_8 = label3 == 778
    # plt.scatter(X_tsne[a3_8, 0], X_tsne[a3_8, 1], c='fuchsia')
    # a3_9 = label3 == 779
    # plt.scatter(X_tsne[a3_9, 0], X_tsne[a3_9, 1], c='pink')
    # a3_10 = label3 == -1
    # plt.scatter(X_tsne[a3_10, 0], X_tsne[a3_10, 1], c='#808080')
    # # plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=label3)
    # plt.xticks([])
    # plt.yticks([])

    # plt.savefig('/home/wpf/ucf/%d.jpg'%i)
    # plt.show()