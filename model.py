import tensorflow as tf
from tensorflow.keras import Model
from tf.keras.layers import Dense, LSTM, Bidirectional, BatchNormalization, Dropout, Lambda
from attention import SimpleAttention, SelfAttention


class AttentiveLSTM(Model):
    """
    Self-attentive LSTM based on Coskun et al. (2018)
    
    Currently with triplet semihard loss as MMD-NCA is not implemented yet.
    """

    def __init__(self, hidden_units=128, r=5, d=10, classification=False, **kwargs):
        """
        Constructor
        
        Args:
            hidden_units (int): number of hidden units for the LSTM, i.e. size of embedding
            r (int): number of frames in the motion sequence which the model should "pay attention" to
            d (int): determines size of attention layer weight matrices
            classification (Bool): whether to include a top layer for classification
        
        Keyword args:
            nr_classes (int): number of classes for classification task. Determines number of units in top layer.
                Only necessary when classification==True
        """
        super(AttentiveLSTM, self).__init__()

        self.classification = classification

        self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.0001,
            decay_steps=50,
            decay_rate=0.96
        )
        self.optimizer = tf.keras.optimizers.SGD(
            learning_rate=self.lr_schedule, 
            momentum=0.9,
            global_clipnorm=25
            )
        self.dropout_rate = 0.5

        self.bi_lstm = Bidirectional(LSTM(units=hidden_units, return_sequences=True))
        # output of shape seq_len x hidden_units*2

        self.batchnorm1 = BatchNormalization()
        self.dropout1 = Dropout(rate=self.dropout_rate)

        # attention layer
        self.selfAttention = SelfAttention(r=r, d=d, lstm_units=hidden_units)

        self.batchnorm2 = BatchNormalization()
        self.dropout2 = Dropout(rate=self.dropout_rate)
        self.dense1 = Dense(units=320, activation='leaky_relu')

        self.batchnorm3 = BatchNormalization()
        self.dropout3 = Dropout(rate=self.dropout_rate)
        self.dense2 = Dense(units=320, activation='leaky_relu')

        self.batchnorm4 = BatchNormalization()
        self.dense3 = Dense(units=128) #activation?

        # l2 normalization
        self.l2 = Lambda(lambda x: tf.math.l2_normalize(x, axis=-1))

        # flatten for classification
        if self.classification==True:
            self.flatten = tf.keras.layers.Flatten()
            self.out = Dense(units=kwargs['nr_classes'], activation='softmax')


    def call(self, input, training=False):
        """
        Feed input through network.
        
        Args:
            input (tensor): input to the model. Expects 3D tensor of form (batch, timesteps, feature)
            training (Bool): whether to use training or inference mode. Default: True (inference), set to False for training
        Returns:
            x: output of the model
        """
        x = self.bi_lstm(input)
        
        x = self.batchnorm1(x, training=training)
        x = self.dropout1(x, training=training)

        x = self.selfAttention(x)

        x = self.batchnorm2(x, training=training)
        x = self.dropout2(x, training=training)
        x = self.dense1(x)
        
        x = self.batchnorm3(x, training=training)
        x = self.dropout3(x, training=training)
        x = self.dense2(x)

        x = self.batchnorm4(x, training=training)
        x = self.dense3(x)

        x = self.l2(x)

        if self.classification == True:
            x = self.flatten(x)
            x = self.out(x)

        return x