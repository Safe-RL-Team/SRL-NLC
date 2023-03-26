import h5py
import tensorflow as tf
from transformers import BertTokenizer


class generator:
    def __init__(self, file, description_size, mission_key="missions"):
        self.file = file
        self.description_size = description_size
        self.mission_key = mission_key
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    def __call__(self):
        with h5py.File(self.file, 'r') as hf:
            for e in range(hf["obs"].shape[0]):
                token = self.tokenizer(hf[self.mission_key][e].decode("utf-8"),
                                       return_tensors="tf",
                                       padding="max_length",
                                       max_length=self.description_size)
                yield (tf.reshape(token['input_ids'], [-1]),
                       tf.reshape(token['token_type_ids'], [-1]),
                       tf.reshape(token['attention_mask'], [-1]),
                       hf["obs"][e]), hf["constraint_mask"][e]

    def len(self):
        with h5py.File(self.file, 'r') as hf:
            return hf["obs"].shape[0]
