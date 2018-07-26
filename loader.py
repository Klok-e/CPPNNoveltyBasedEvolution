import numpy as np


def __unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def __load_batch(i: int) -> dict:
    batches = [("data_batch_" + str(i)) for i in range(1, 6)]
    return __unpickle("cifar-10-batches-py/" + batches[i])


def __cifar10_to_img(img_flat):
    import numpy as np
    img_R = img_flat[0:1024].reshape((32, 32))
    img_G = img_flat[1024:2048].reshape((32, 32))
    img_B = img_flat[2048:3072].reshape((32, 32))
    img = np.dstack((img_R, img_G, img_B))
    return img


def save_img(img, filename):
    import PIL.Image
    i = PIL.Image.fromarray(img, "RGB")
    i.save(filename + ".png")


def get_images_batch(i: int):
    images_and_labels = __load_batch(i)
    images = [__cifar10_to_img(images_and_labels[b"data"][i]) for i in range(len(images_and_labels[b"data"]))]
    return images


def normalize(images):
    for i in range(len(images)):
        images[i] = (images[i] / 255).astype(np.float32)


def resize_from_32x32x3_to_3x32x32(image):
    return np.transpose(image, (2, 0, 1))


def resize_from_3x32x32_to_32x32x3(image):
    return np.transpose(image, (1, 2, 0))
