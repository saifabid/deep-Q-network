import os
from collections import defaultdict

from common.base_model import BaseModel
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_datasets as tfds


class SimpleNN(BaseModel, tf.keras.Model):
    checkpoint_directory = "/tmp/training_checkpoints/simplenn"
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")

    def __init__(self, observation_shape, action_shape):
        super(SimpleNN, self).__init__("simple_nn")

        self.observation_shape = observation_shape
        self.action_shape = action_shape

        self.define_layers()
        self.optimizer = SimpleNN.get_optimizer()
        self.loss_object = SimpleNN.get_loss_object()

        self.metrics = self.metrics()

    def define_layers(self):
        self.dense1 = layers.Dense(self.observation_shape, activation=tf.nn.relu)
        self.dense2 = layers.Dense(32, activation=tf.nn.relu)
        self.dense3 = layers.Dense(16, activation=tf.nn.relu)
        self.predictions = layers.Dense(self.action_shape, activation=tf.nn.softmax)

    def reset_metrics(self):
        for _, v in self.metrics.items():
            for _, vv in v.items():
                vv.reset_states()

    @staticmethod
    def metrics():
        metrics = defaultdict(dict)
        metrics['train']['loss'] = tf.keras.metrics.Mean(name='train_loss')
        metrics['train']['accuracy'] = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        metrics['test']['loss'] = tf.keras.metrics.Mean(name='test_loss')
        metrics['test']['accuracy'] = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

        return metrics

    @staticmethod
    def get_optimizer():
        return tf.optimizers.Adam()

    @staticmethod
    def get_loss_object():
        return tf.losses.SparseCategoricalCrossentropy()

    @tf.function
    def call(self, inputs):  # tf.keras.Model calls call() internally when the object is called
        fprop = self.dense1(inputs)
        fprop = self.dense2(fprop)
        fprop = self.dense3(fprop)

        return self.predictions(fprop)

    def train(self, data):
        for x, y in data:
            with tf.GradientTape() as tape:
                y_ = self(x)
                loss = self.loss_object(y, y_)
            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

            self.metrics['train']['loss'](loss)
            self.metrics['train']['accuracy'](y, y_)

    def test(self, data):
        for x, y in data:
            y_ = self(x)
            t_loss = self.loss_object(y, y_)

            self.metrics['test']['loss'](t_loss)
            self.metrics['test']['accuracy'](y, y_)

    def forward(self, inputs):
        return self.call(inputs)

    def save_model(self, *args, **kwargs):
        checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self)
        checkpoint.save(file_prefix=SimpleNN.checkpoint_prefix)

    def load(self, *args, **kwargs):
        checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self)
        status = checkpoint.restore(tf.train.latest_checkpoint(SimpleNN.checkpoint_directory))
        print("Status: ", status)


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

    #model.load()

    for epoch in range(EPOCHS):
        model.train(mnist_train)
        model.test(mnist_test)
        template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print(template.format(epoch + 1,
                              model.metrics['train']['loss'].result(),
                              model.metrics['train']['accuracy'].result() * 100,
                              model.metrics['test']['loss'].result(),
                              model.metrics['test']['accuracy'].result() * 100))
        model.save_model()
