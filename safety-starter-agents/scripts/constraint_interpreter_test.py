import tensorflow as tf

from generator import generator
from ConstraintMaskModule import *
from transformers import BertTokenizer

def loss(label, pred):
    # mask = label != 0
    loss_object = tf.keras.losses.BinaryCrossentropy(axis=1,
                                                     from_logits=True, reduction="sum_over_batch_size")
    loss = loss_object(label, pred)

    # mask = tf.cast(mask, dtype=loss.dtype)
    # loss *= mask
    # print(loss.shape)
    # print(label.shape)
    # loss = tf.reduce_sum(loss) / label.shape[1]
    return loss


def accuracy(label, pred):
    # pred = tf.argmax(pred, axis=2)
    # accuracy_func = tf.keras.metrics.BinaryAccuracy(name='train_accuracy', threshold=0.)
    pred = tf.cast(pred > 0, tf.int32)
    label = tf.cast(label, pred.dtype)
    match = tf.cast(label == pred, tf.int32)

    # mask = label != 0
    #
    # match = match & mask

    # match = tf.cast(match, dtype=tf.float32)
    # mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(match) / (BATCH_SIZE * label.shape[1])

if __name__ == '__main__':
    BATCH_SIZE = 50
    DESCRIPTION_SIZE = 100
    D_MODEL = 16

    out_types = ((tf.int32, tf.int32, tf.int32, tf.float32), tf.int32)
    out_shapes = (((DESCRIPTION_SIZE,), (DESCRIPTION_SIZE,), (DESCRIPTION_SIZE,), (7, 7, 3,)), (49,))

    test_generator = generator("../data/mask/test.h5", DESCRIPTION_SIZE, "missions_paraphrased")
    test_batches = int(test_generator.len() / BATCH_SIZE)
    test_data = tf.data.Dataset.from_generator(generator=test_generator,
                                               output_types=out_types,
                                               output_shapes=out_shapes).batch(BATCH_SIZE)

    learning_rate = CustomSchedule(8 * D_MODEL)

    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                         epsilon=1e-9)

    model = ConstraintMaskModule(obs_size=7,
                                 obs_channels=3,
                                 description_size=DESCRIPTION_SIZE,
                                 conv_filter=10,
                                 kernel_size=3,
                                 num_layers=1,
                                 d_model=D_MODEL,
                                 num_heads=1,
                                 dff=256,
                                 vocab_size=50)

    test_constraint = ["test" for i in range(BATCH_SIZE)]
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    inputs = tokenizer(test_constraint, return_tensors="tf", padding="max_length", max_length=DESCRIPTION_SIZE)

    test_obs = tf.zeros(
        (BATCH_SIZE, 7, 7, 3),
        dtype=tf.dtypes.float32,
        name='obs'
    )

    output = model((inputs['input_ids'], inputs['token_type_ids'], inputs['attention_mask'], test_obs))

    checkpoint_path = "./dummy/cp-0006.ckpt"
    model.load_weights(checkpoint_path)

    model.summary()

    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=[accuracy])

    print("Evaluate on test data")
    results = model.evaluate(test_data)
    print("test loss, test acc:", results)
