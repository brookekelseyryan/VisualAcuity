import PIL
from PIL import ImageOps
from tensorflow.keras.preprocessing.image import img_to_array


def augment(image, file_title):
    """
    Augments a given image.
    :return Dictionary: key - file title to append, value - image as numpy array
    """
    return {file_title + "_gray": img_to_array(PIL.ImageOps.grayscale(image)),
            file_title + "_120_bright": img_to_array(PIL.ImageEnhance.Brightness(image).enhance(1.2)),
            file_title + "_120_contrast": img_to_array(PIL.ImageEnhance.Contrast(image).enhance(1.2)),
            file_title + "_80_bright": img_to_array(PIL.ImageEnhance.Brightness(image).enhance(0.8)),
            file_title + "_80_contrast": img_to_array(PIL.ImageEnhance.Contrast(image).enhance(0.8))}
