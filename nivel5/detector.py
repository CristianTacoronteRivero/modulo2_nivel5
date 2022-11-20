from tensorflow.keras.applications.resnet50 import ResNet50
from keras.utils import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import matplotlib.pyplot as plt
import numpy as np

class DetectorResNet50:
    def __init__(self, weights='imagenet'):
        self.model = ResNet50(weights=weights)

    def evaluate(self, img_fname, n_pred:int = 5, show_image:bool = False):
        """Funcion encargada de predecir que elemento se encuentra en una imagen

        :param img_fname: Imagen con formato .jpg
        :type img_fname: Imagen
        :param n_pred: Numero de clases que se pretende predecir, por defecto 5
        :type n_pred: int, optional
        :param show_image: True si se desea mostrar en un notebook la imagen defaults to True
        :type show_image: bool, optional
        """
        image = load_img(img_fname, target_size=(224,224))
        x = img_to_array(image)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        preds = self.model.predict(x)

        print(f"Clases predichas: {decode_predictions(preds, top=n_pred)[0]}")

        if show_image:
            plt.imshow(image)

if __name__ == '__main__':
    dt = DetectorResNet50()
    dt.evaluate('nivel5/chiuaua.jpeg', n_pred=3)