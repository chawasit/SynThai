import csv
import gc
import glob
import json
import os
import shutil
import sys
import warnings
from collections import Counter
from datetime import datetime
from multiprocessing import Process, Queue
from pprint import pprint

from keras.models import load_model

import fire
import numpy as np
from sklearn.exceptions import UndefinedMetricWarning

from . import constant
from .callback import CustomCallback
from .metric import custom_metric
from .model import Model
from .utils import Corpus, InputBuilder, DottableDict, index_builder

from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))


class SynThai(object):
    def __init__(self, model_path, model_num_step):
        self.char_index = index_builder(constant.CHARACTER_LIST, constant.CHAR_START_INDEX)
        self.tag_index = index_builder(constant.TAG_LIST, constant.TAG_START_INDEX)
        print("Loading Model")
        self.model = load_model(model_path)
        self.model_num_step = model_num_step
        print("Loaded")

    def tokenize(self, input_texts, word_delimiter="|", tag_delimiter="/"):
        # Load text
        texts = Corpus()
        texts.add_text(input_texts)

        # Generate input
        inb = InputBuilder(texts, self.char_index, self.tag_index, self.model_num_step,
                            text_mode=True)

        # Run on each text
        for text_idx in range(texts.count):
            # Get character list and their encoded list
            x_true = texts.get_char_list(text_idx)
            encoded_x = inb.get_encoded_char_list(text_idx)

            # Predict
            y_pred = self.model.predict(encoded_x)
            y_pred = np.argmax(y_pred, axis=2)

            # Flatten to 1D
            y_pred = y_pred.flatten()

            # Result list
            result = list()

            # Process on each character
            for idx, char in enumerate(x_true):
                # Character label
                label = y_pred[idx]

                # Pad label
                if label == constant.PAD_TAG_INDEX:
                    continue

                # Append character to result list
                result.append(char)

                # Skip tag for spacebar character
                if char == constant.SPACEBAR:
                    continue

                # Tag at segmented point
                if label != constant.NON_SEGMENT_TAG_INDEX:
                    # Index offset
                    index_without_offset = label - constant.TAG_START_INDEX

                    # Tag name
                    tag_name = constant.TAG_LIST[index_without_offset]

                    # Append delimiter and tag to result list
                    result.append(tag_delimiter)
                    result.append(tag_name)
                    result.append(word_delimiter)

        return "".join(result)
