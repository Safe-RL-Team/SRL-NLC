import numpy as np
import h5py
import random

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def make_synonym_mission(fpath):
    # tokenizer = AutoTokenizer.from_pretrained("t5-base")
    # model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

    epoch_len = 100

    tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")
    model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws").to("cuda")

    f = h5py.File(fpath + "/test" + '.h5', 'r+')
    print(f["missions"].shape)

    paraphrase = []
    for i in range(int(f["missions"].shape[0] / epoch_len)):
        index = i * epoch_len
        print("i: " + str(i))
        # sentence = "paraphrase: " + f["missions"][index][0].decode("utf-8") + " </s>"
        sentence = f["missions"][index].decode("utf-8")
        print("original constraint: " + sentence)
        encoding = tokenizer.encode_plus(sentence, padding=True, return_tensors="pt")
        input_ids, attention_masks = encoding["input_ids"].to("cuda"), encoding["attention_mask"].to("cuda")

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_masks,
            max_length=256,
            do_sample=True,
            top_k=120,
            top_p=0.95,
            early_stopping=True,
            num_return_sequences=1)

        output = tokenizer.decode(outputs[0],
                                  skip_special_tokens=True,
                                  clean_up_tokenization_spaces=True)
        paraphrase.extend([output for i in range(epoch_len)])
        print("Paraphrase: " + output)

    f.create_dataset('missions_paraphrased', (len(paraphrase)), dtype=h5py.special_dtype(vlen=str), data=paraphrase)
    f.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fpath', type=str, default="../data/mask")
    args = parser.parse_args()
    make_synonym_mission(args.fpath)

    with h5py.File(args.fpath + "/test" + '.h5', 'r') as f:
        for key in f.keys():
            print(f[key], key, f[key].name)