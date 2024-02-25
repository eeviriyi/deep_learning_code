import random
import tensorflow as tf


def synthetic_data(w, b, num):
    """Generate a data set that obeys y = Xw + b + noise

    :w
    :b
    :num: the number of the data set
    :returns: a data set
    """
    X = tf.zeros((num, w.shape[0]))
    X += tf.random.normal(shape=X.shape)
    y = tf.matmul(X, tf.reshape(w, (-1, 1))) + b
    y += tf.random.normal(shape=y.shape, stddev=0.01)
    y = tf.reshape(y, (-1, 1))
    return X, y


def data_iter(batch_size, features, labels):
    """Randomly yield the data set by batch_size

    :batch_size:
    :features:
    :labels:
    :yield:
    """
    num = len(features)
    indices = list(range(num))
    random.shuffle(indices)
    for i in range(0, num, batch_size):
        j = tf.constant(indices[i : min(i + batch_size, num)])
        yield tf.gather(features, j), tf.gather(labels, j)


# 定义模型：线性模型
def linreg(X, w, b):
    return tf.matmul(X, w) + b


# 定义损失函数：平方损失
def squared_loss(y_hat, y):
    return (y_hat - tf.reshape(y, y_hat.shape)) ** 2 / 2


# 定义优化算法：小批量随机梯度下降
def sgd(params, grads, lr, batch_size):
    for param, grad in zip(params, grads):
        param.assign_sub(lr * grad / batch_size)


# 训练
if __name__ == "__main__":

    true_w = tf.constant([2, -3.4])
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, 1000)

    # 初始化模型参数
    w = tf.Variable(tf.random.normal(shape=(2, 1), mean=0, stddev=0.01), trainable=True)
    b = tf.Variable(tf.zeros(1), trainable=True)

    lr = 0.03
    num_epochs = 3
    net = linreg
    loss = squared_loss
    batch_size = 10

    for epoch in range(num_epochs):
        for X, y in data_iter(batch_size, features, labels):
            with tf.GradientTape() as g:
                l = loss(net(X, w, b), y)
            dw, db = g.gradient(l, [w, b])
            sgd([w, b], [dw, db], lr, batch_size)
        train_l = loss(net(features, w, b), labels)
        print(f"epoch {epoch + 1}, loss {float(tf.reduce_mean(train_l)):f}")
