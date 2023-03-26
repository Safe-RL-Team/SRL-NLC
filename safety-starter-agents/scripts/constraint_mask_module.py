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


# @tf.function
def train_step(model, constraint_code, obs, constraint_mask, loss_obj_func, train_loss_func, train_accuracy_func):
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model((constraint_code, obs), training=True)
        loss = loss_obj_func(constraint_mask, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss_func(loss)
    train_accuracy_func(constraint_mask, predictions)


@tf.function
def test_step(model, obs, constraint_mask, constraint_code, loss_obj_func, train_loss_func, train_accuracy_func):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model((constraint_code, obs), training=False)
    t_loss = loss_obj_func(constraint_mask, predictions)

    train_loss_func(t_loss)
    train_accuracy_func(constraint_mask, predictions)


if __name__ == '__main__':
    EPOCHS = 5
    BATCH_SIZE = 50
    description_size = 100


    def dataset_fn(input, label):
        dim = tf.reduce_prod(tf.shape(label)[1:])
        label = tf.reshape(label, [-1, dim])
        return input, label


    out_types = ((tf.int32, tf.int32, tf.int32, tf.float32), tf.int32)
    out_shapes = (((description_size, ), (description_size, ), (description_size, ), (7, 7, 3,)), (49,))

    train_generator = generator("../data/mask/train.h5", description_size, "mission")
    train_data = tf.data.Dataset.from_generator(generator=train_generator,
                                                output_types=out_types,
                                                output_shapes=out_shapes).batch(BATCH_SIZE)

    batches = int(train_generator.len() / BATCH_SIZE)
    train_data = train_data.apply(tf.data.experimental.assert_cardinality(batches))
    # train_data = train_data.map(dataset_fn)

    test_generator = generator("../data/mask/test.h5", description_size, "mission")
    test_batches = int(test_generator.len() / BATCH_SIZE)
    test_data = tf.data.Dataset.from_generator(generator=test_generator,
                                               output_types=out_types,
                                               output_shapes=out_shapes).batch(BATCH_SIZE)
    # test_data = test_data.map(dataset_fn)

    for (input_ids, token_type_ids, attention_mask, o), label in train_data:
        print(input_ids.shape)
        print(token_type_ids.shape)
        print(attention_mask.shape)
        print(o.shape)
        print(label.shape)
        break

    d_model = 16
    model = ConstraintMaskModule(obs_size=7,
                                 obs_channels=3,
                                 description_size=description_size,
                                 conv_filter=10,
                                 kernel_size=3,
                                 num_layers=1,
                                 d_model=d_model,
                                 num_heads=1,
                                 dff=256,
                                 vocab_size=50)

    learning_rate = CustomSchedule(8 * d_model)

    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                         epsilon=1e-9)

    # test_constraint = tf.zeros(
    #     (batch_size, 10),
    #     dtype=tf.dtypes.float32,
    #     name='constraint'
    # )

    test_constraint = ["test" for i in range(BATCH_SIZE)]
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    inputs = tokenizer(test_constraint, return_tensors="tf", padding="max_length", max_length=description_size)

    test_obs = tf.zeros(
        (BATCH_SIZE, 7, 7, 3),
        dtype=tf.dtypes.float32,
        name='obs'
    )

    output = model((inputs['input_ids'], inputs['token_type_ids'], inputs['attention_mask'], test_obs))

    # checkpoint_path = "./model2/cp-0003.ckpt"
    # model.load_weights(checkpoint_path)

    # print(test_constraint.shape)
    print(test_obs.shape)
    print(output.shape)

    model.summary()

    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=[accuracy])

    checkpoint_path = "./dummy/cp-{epoch:04d}.ckpt"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path, verbose=1, save_weights_only=True,
        # Save weights, every epoch.
        save_freq='epoch')

    model.fit(train_data,
              epochs=100,
              validation_data=test_data, callbacks=[cp_callback])

    # loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    # optimizer = tf.keras.optimizers.Adam()
    #
    # train_loss = tf.keras.metrics.Mean(name='train_loss')
    # train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy', threshold=0.)
    #
    # test_loss = tf.keras.metrics.Mean(name='test_loss')
    # test_accuracy = tf.keras.metrics.BinaryAccuracy(name='test_accuracy', threshold=0.)
    #
    # for epoch in range(EPOCHS):
    #     # Reset the metrics at the start of the next epoch
    #     train_loss.reset_states()
    #     train_accuracy.reset_states()
    #     test_loss.reset_states()
    #     test_accuracy.reset_states()
    #
    #     for batch_num, (obs, constraint_mask, constraint_code, hc) in enumerate(train_data):
    #         print("Training epoch: " + str(epoch) + " batch: " + str(batch_num) + "/" + str(batches))
    #         train_step(model, constraint_code, obs, constraint_mask, loss_object, train_loss, test_loss)
    #         if batch_num > 5:
    #             break
    #
    #     print(
    #         f'Epoch {epoch + 1}, '
    #         f'Loss: {train_loss.result()}, '
    #         f'Accuracy: {train_accuracy.result() * 100}, '
    #         f'Test Loss: {test_loss.result()}, '
    #         f'Test Accuracy: {test_accuracy.result() * 100}'
    #     )
    #     print("Saving model...")
    #     tf.saved_model.save(model, export_dir='./model')
    #
    # for batch_num, (obs, constraint_mask, constraint_code, hc) in enumerate(test_data):
    #     print("Testing: batch: " + str(batch_num) + "/" + str(test_batches))
    #     test_step(model, obs, constraint_mask, constraint_code, loss_object, train_loss, test_loss)
    #     if batch_num > 5:
    #         break
