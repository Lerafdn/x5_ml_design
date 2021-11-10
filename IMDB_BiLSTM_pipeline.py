from metaflow import FlowSpec, step, retry, catch, batch, IncludeFile, Parameter, conda, conda_base


def get_python_version():
    """
    A convenience function to get the python version used to run this
    tutorial. This ensures that the conda environment is created with an
    available version of python.
    """
    import platform
    versions = {'3': '3.8'}
    return versions[platform.python_version_tuple()[0]]


# Use the specified version of python for this flow.
@conda_base(python=get_python_version())
class IMDB_BiLSTM(FlowSpec):

    @conda(libraries={'tensorflow': '2.6.0'})
    @step
    def start(self):
        import numpy as np
        from tensorflow import keras
        from tensorflow.keras import layers
        import logging
        self.max_features = 20000  # Only consider the top 20k words
        self.maxlen = 200  # Only consider the first 200 words of each movie review
        logging.info('start finished')
        self.next(self.init_neural_network)

    @retry(times=1)
    @conda(libraries={'tensorflow': '2.6.0'})
    @step
    def init_neural_network(self):
        import numpy as np
        from tensorflow import keras
        from tensorflow.keras import layers
        import logging
        # Input for variable-length sequences of integers
        inputs = keras.Input(shape=(None,), dtype="int32")
        # Embed each integer in a 128-dimensional vector
        x = layers.Embedding(self.max_features, 128)(inputs)
        # Add 2 bidirectional LSTMs
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
        x = layers.Bidirectional(layers.LSTM(64))(x)
        # Add a classifier
        outputs = layers.Dense(1, activation="sigmoid")(x)
        self.model = keras.Model(inputs, outputs)
        self.model.summary()
        logging.info('init_neural_network finished')
        self.next(self.load_data)

    @retry(times=1)
    @conda(libraries={'tensorflow': '2.6.0'})
    @step
    def load_data(self):
        import numpy as np
        from tensorflow import keras
        from tensorflow.keras import layers
        import logging
        (self.x_train, self.y_train), (self.x_val, self.y_val) = keras.datasets.imdb.load_data(
            num_words=self.max_features)

        print(len(self.x_train), "Training sequences")
        print(len(self.x_val), "Validation sequences")

        # Use pad_sequence to standardize sequence length:
        # this will truncate sequences longer than 200 words and zero-pad sequences shorter than 200 words.
        self.x_train = keras.preprocessing.sequence.pad_sequences(self.x_train, maxlen=self.maxlen)
        self.x_val = keras.preprocessing.sequence.pad_sequences(self.x_val, maxlen=self.maxlen)
        logging.info('load_data finished')
        self.next(self.train)

    @retry(times=1)
    @conda(libraries={'tensorflow': '2.6.0'})
    @step
    def train(self):
        import logging
        self.model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        self.model.fit(self.x_train, self.y_train, batch_size=32, epochs=2, validation_data=(self.x_val, self.y_val))
        logging.info('train finished')
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == '__main__':
    IMDB_BiLSTM()
