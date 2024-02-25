import tensorflow as tf

from linear_regression_scratch import synthetic_data


def load_array(data_arrays, batch_size, is_train=True):
    """A data set generator"""
    dataset = tf.data.Dataset.from_tensor_slices(data_arrays)
    if is_train:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    return dataset


if __name__ == "__main__":
    # Generate data set
    true_w = tf.constant([2, -3.4])
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, 1000)

    # Read data set
    batch_size = 10
    data_iter = load_array((features, labels), batch_size)

    # Define the model
    # initializer = tf.initializers.RandomNormal(stddev=0.01)
    net = tf.keras.Sequential()
    net.add(tf.keras.layers.Dense(1, kernel_initializer='random_uniform'))

    # Define the loss function
    loss = tf.keras.losses.MeanSquaredError()

    # Define the optimization algorithm
    trainer = tf.keras.optimizers.SGD(learning_rate=0.03)

    # Train
    num_epochs = 3
    for epoch in range(num_epochs):
        for X, y in data_iter:
            with tf.GradientTape() as tape:
                l = loss(net(X, training=True), y)
            grads = tape.gradient(l, net.trainable_variables)
            trainer.apply_gradients(zip(grads, net.trainable_variables))
        l = loss(net(features), labels)
        print(f"epoch {epoch + 1}, loss {l:f}")

    w = net.get_weights()[0]
    print('w的估计误差：', true_w - tf.reshape(w, true_w.shape))
    b = net.get_weights()[1]
    print('b的估计误差：', true_b - b)
