import numpy as np
import cv2
import matplotlib.pyplot as plt;


def draw_network_inputs(images):
    plt.imshow(np.transpose(images.tensors[0].cpu().detach().numpy()/255 + 0.5, (1, 2, 0)))
    plt.show()

    plt.imshow(np.transpose(images.tensors[1].cpu().detach().numpy() / 255 + 0.5, (1, 2, 0)))
    plt.show()

    plt.imshow(np.transpose(images.tensors[2].cpu().detach().numpy() / 255 + 0.5, (1, 2, 0)))
    plt.show()


def plot_inference_for_image(predictor, image_path):
    image = cv2.imread(image_path)
    result = predictor.run_on_opencv_image(image)

    plt.imshow(result)
    plt.show()