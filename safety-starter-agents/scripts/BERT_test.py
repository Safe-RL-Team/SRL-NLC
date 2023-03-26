from transformers import BertTokenizer, TFBertModel
import tensorflow as tf

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = TFBertModel.from_pretrained('bert-base-cased', return_dict=True)

inputs = tokenizer("Hello, my dog is cute. I like him.", return_tensors="tf")
inputs_padded = tokenizer("Hello, my dog is cute. I like him.", return_tensors="tf", padding="max_length", max_length = 100)
outputs = model(inputs_padded)
print(outputs.last_hidden_state.shape)