import numpy as np
import logging, sys, json
import timeit as t
import base64
from PIL import Image, ImageOps
from io import BytesIO
import keras
import os

MODEL_FILE = 'classifier.h5'

logger = logging.getLogger("ai_logger")
ch = logging.StreamHandler(sys.stdout)
logger.addHandler(ch)


trainedModel = None
mem_after_init = None


def init():
    """ Initialise model
    """
    global trainedModel, mem_after_init

    start = t.default_timer()
    trainedModel = keras.models.load_model(MODEL_FILE)
    end = t.default_timer()

    loadTimeMsg = "Model loading time: {0} ms".format(round((end - start) * 1000, 2))
    logger.info(loadTimeMsg)

# Removed argument from run
def run():
    """ Classify the input using the loaded model
    """
    filepath_to_test = 'test_images/'

    start = t.default_timer()

    #images = inputString
    result = []
    files = []
    totalPreprocessTime = 0
    totalEvalTime = 0
    totalResultPrepTime = 0

    list_of_files = os.listdir(filepath_to_test)
    list_of_test_images = []
    
    for file in list_of_files:
        filepath = filepath_to_test + file
        list_of_test_images.append(filepath)
   
    for image in list_of_test_images:
        files.append(image)
        with open(image, 'rb') as image_file:
            inputString = base64.b64encode(image_file.read())
        if inputString.startswith(b'b\''):
            inputString = inputString[2:-1]
    #     base64Img = base64ImgString.encode('utf-8')

    #     # Preprocess the input data
        startPreprocess = t.default_timer()
        decoded_img = base64.b64decode(inputString)
        img_buffer = BytesIO(decoded_img)

    #     # Load image with PIL (RGB)
        pil_img = Image.open(img_buffer).convert('RGB')


        pil_img = ImageOps.fit(pil_img, (64, 64), Image.ANTIALIAS)
        rgb_image = np.array(pil_img, dtype=np.float32)

        from keras.preprocessing import image

        pil_img = image.img_to_array(pil_img)
        pil_img = np.expand_dims(pil_img, axis=0)

        endPreprocess = t.default_timer()
        totalPreprocessTime += endPreprocess - startPreprocess

        # Evaluate the model using the input data
        startEval = t.default_timer()
        img_result = trainedModel.predict(pil_img)
        if img_result[0][0] == 1:
            result.append('dog')
        else:
            result.append('cat')
        endEval = t.default_timer()
        totalEvalTime += endEval - startEval

        #except:
         #   result.append('ERROR: Evaluation not successful')

    end = t.default_timer()

    logger.info("Predictions: {0}".format(result))
    logger.info("Predictions took {0} ms".format(round((end - start) * 1000, 2)))
    logger.info("Time distribution: preprocess={0} ms, eval={1} ms, resultPrep = {2} ms".format(
        round(totalPreprocessTime * 1000, 2), round(totalEvalTime * 1000, 2), round(totalResultPrepTime * 1000, 2)))

    actualWorkTime = round((totalPreprocessTime + totalEvalTime + totalResultPrepTime) * 1000, 2)
    return (files, result, 'Computed in {0} ms'.format(actualWorkTime))