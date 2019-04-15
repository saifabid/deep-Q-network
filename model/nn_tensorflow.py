from common.base_model import BaseModel
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_datasets as tfds


class SimpleNN(BaseModel, tf.keras.Model):
    def __init__(self, observation_shape, action_shape):
        super(SimpleNN, self).__init__("simple_nn")

        self.dense_1 = layers.Dense(observation_shape, activation=tf.nn.relu)
        self.dense_2 = layers.Dense(32, activation=tf.nn.relu)
        self.dense_3 = layers.Dense(16, activation=tf.nn.relu)
        self.predictions = layers.Dense(action_shape, activation=tf.nn.softmax)

        self.optimizer = SimpleNN.get_optimizer()
        self.loss_object = SimpleNN.tf.losses.SparseCategoricalCrossentropy()

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    def reset(self):
        pass

    @staticmethod
    def get_optimizer():
        return tf.optimizers.Adam()

    @staticmethod
    def get_loss_object():
        return tf.losses.SparseCategoricalCrossentropy()

    def call(self, inputs):  # tf.keras.Model calls call() internally when the object is called
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        x = self.dense_3(x)
        x = self.predictions(x)

        return x

    def train(self, data):
        for x, y in data:
            with tf.GradientTape() as tape:
                y_ = self(x)
                loss = self.loss_object(y, y_)
            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

            self.train_loss(loss)
            self.train_accuracy(y, y_)

    def test(self, data):
        for x, y in data:
            y_ = self(x)
            t_loss = self.loss_object(y, y_)

            self.test_loss(t_loss)
            self.test_accuracy(y, y_)

    def forward(self, inputs):
        return self.call(inputs)

    def saveModel(self, *args, **kwargs):
        pass

    def loadModel(self, *args, **kwargs):
        pass


if __name__ == '__main__':  # train mnist
    dataset, info = tfds.load('mnist', data_dir='gs://tfds-data/datasets', with_info=True, as_supervised=True,
                              download=True)
    mnist_train, mnist_test = dataset['train'], dataset['test']


    def convert_types(image, label):
        image = tf.cast(image, tf.float32)
        image /= 255
        image = tf.reshape(image, (784,))

        return image, label


    mnist_train = mnist_train.map(convert_types).shuffle(10000).batch(32)
    mnist_test = mnist_test.map(convert_types).batch(32)

    EPOCHS = 15
    model = SimpleNN(784, 10)

    for epoch in range(EPOCHS):
        model.train(mnist_train)
        model.test(mnist_test)
        template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print(template.format(epoch + 1,
                              model.train_loss.result(),
                              model.train_accuracy.result() * 100,
                              model.test_loss.result(),
                              model.test_accuracy.result() * 100))
