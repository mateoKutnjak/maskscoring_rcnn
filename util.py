import numpy as np
import matplotlib.pyplot as plt;


def draw_network_inputs(images):
    plt.imshow(np.transpose(images.tensors[0].cpu().detach().numpy()/255 + 0.5, (1, 2, 0)))
    plt.show()

    plt.imshow(np.transpose(images.tensors[1].cpu().detach().numpy() / 255 + 0.5, (1, 2, 0)))
    plt.show()

    plt.imshow(np.transpose(images.tensors[2].cpu().detach().numpy() / 255 + 0.5, (1, 2, 0)))
    plt.show()