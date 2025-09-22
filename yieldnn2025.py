import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras


def crop_conc(x1, x2):
    crop_x2 = x2[:, : tf.shape(x1)[1], :]
    return tf.concat([x1, crop_x2], axis=2)


class YieldNN(tf.keras.Model):
    def __init__(self, MaxPoolRate=[8, 6, 4, 2], input_s=(9600, 4), **kwargs):
        super(YieldNN, self).__init__(**kwargs)
        self.MaxPoolRate = MaxPoolRate
        self.kernel_size = 9

        self.enc_1 = tf.keras.Sequential(
            [
                layers.Conv1D(filters=16, kernel_size=self.kernel_size, padding="same", input_shape=input_s),
                layers.BatchNormalization(),
                layers.ReLU(),
            ]
        )

        self.enc_2 = tf.keras.Sequential(
            [
                layers.MaxPooling1D(self.MaxPoolRate[0]),
                layers.Conv1D(filters=32, kernel_size=self.kernel_size, padding="same"),
                layers.BatchNormalization(),
                layers.ReLU(),
            ]
        )

        self.enc_3 = tf.keras.Sequential(
            [
                layers.MaxPooling1D(self.MaxPoolRate[1]),
                layers.Conv1D(filters=64, kernel_size=self.kernel_size, padding="same"),
                layers.BatchNormalization(),
                layers.ReLU(),
            ]
        )

        self.enc_4 = tf.keras.Sequential(
            [
                layers.MaxPooling1D(self.MaxPoolRate[2]),
                layers.Conv1D(filters=128, kernel_size=self.kernel_size, padding="same"),
                layers.BatchNormalization(),
                layers.ReLU(),
            ]
        )

        self.enc_5 = tf.keras.Sequential(
            [
                layers.MaxPooling1D(self.MaxPoolRate[3]),
                layers.Conv1D(filters=256, kernel_size=self.kernel_size, padding="same"),
                layers.BatchNormalization(),
                layers.ReLU(),
            ]
        )

        self.decoder_b1 = tf.keras.Sequential(
            [
                layers.UpSampling1D(self.MaxPoolRate[3]),
                layers.Conv1D(filters=128, kernel_size=self.kernel_size, padding="same"),
                layers.BatchNormalization(),
                layers.ReLU(),
            ]
        )

        self.decoder_b2 = tf.keras.Sequential(
            [
                layers.Conv1D(filters=128, kernel_size=self.kernel_size, padding="same"),
                layers.BatchNormalization(),
                layers.ReLU(),
                layers.UpSampling1D(self.MaxPoolRate[2]),
                layers.Conv1D(filters=64, kernel_size=6, padding="same"),
                layers.BatchNormalization(),
                layers.ReLU(),
            ]
        )

        self.decoder_b3 = tf.keras.Sequential(
            [
                layers.Conv1D(filters=64, kernel_size=self.kernel_size, padding="same"),
                layers.BatchNormalization(),
                layers.ReLU(),
                layers.UpSampling1D(self.MaxPoolRate[1]),
                layers.Conv1D(filters=32, kernel_size=8, padding="same"),
                layers.BatchNormalization(),
                layers.ReLU(),
            ]
        )

        self.decoder_b4 = tf.keras.Sequential(
            [
                layers.Conv1D(filters=32, kernel_size=self.kernel_size, padding="same"),
                layers.BatchNormalization(),
                layers.ReLU(),
                layers.UpSampling1D(self.MaxPoolRate[0]),
                layers.Conv1D(filters=16, kernel_size=10, padding="same"),
                layers.BatchNormalization(),
                layers.ReLU(),
            ]
        )

        self.decoder_b5 = tf.keras.Sequential(
            [
                layers.Conv1D(filters=16, kernel_size=self.kernel_size, padding="same"),
                layers.BatchNormalization(),
                layers.ReLU(),
            ]
        )

        self.lstm_l1 = layers.Bidirectional(layers.LSTM(units=64, return_sequences=True, input_shape=[9600, 16]))

        self.lstm_l2 = layers.Bidirectional(layers.LSTM(units=32, return_sequences=True))

        self.lstm_l3 = layers.LSTM(units=16, return_sequences=True)

        self.segment_classifier = tf.keras.Sequential(
            [
                layers.Conv1D(filters=2, kernel_size=1),
                layers.BatchNormalization(),
                layers.Activation("tanh"),
            ]
        )

        self.final_conv = tf.keras.Sequential([layers.Conv1D(filters=2, kernel_size=1), layers.Softmax(axis=2)])

    def call(self, x):
        enc_1 = self.enc_1(x)
        enc_2 = self.enc_2(enc_1)
        enc_3 = self.enc_3(enc_2)
        enc_4 = self.enc_4(enc_3)
        enc_5 = self.enc_5(enc_4)

        x = self.decoder_b1(enc_5)
        x = crop_conc(x, enc_4)

        x = self.decoder_b2(x)
        x = crop_conc(x, enc_3)

        x = self.decoder_b3(x)
        x = crop_conc(x, enc_2)

        x = self.decoder_b4(x)
        x = crop_conc(x, enc_1)

        x = self.decoder_b5(x)

        x = self.lstm_l1(x)
        x = self.lstm_l2(x)
        x = self.lstm_l3(x)

        x = self.segment_classifier(x)
        x = tf.reshape(x, (-1, 96, 100, 2))
        x = tf.reduce_mean(x, axis=2)
        x = self.final_conv(x)

        return x


# Example usage
if __name__ == "__main__":
    batch_size = 2
    model = YieldNN()
    x = tf.random.uniform((batch_size, 9600, 6))
    y = model(x)
    print(y.shape)
    print(f"loss is {tf.keras.losses.categorical_crossentropy(y, y)}")
    print("Total number of parameters is: {}".format(model.count_params()))
