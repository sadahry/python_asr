# 以下を踏襲 https://keras.io/examples/vision/captcha_ocr/
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class CTCLayer(layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_pred, labels, feat_lens, label_lens):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.
        batch_len = tf.cast(tf.shape(labels)[0], dtype=tf.int32)
        feat_lens = feat_lens * tf.ones(shape=(batch_len, 1), dtype=tf.int32)
        label_lens = label_lens * tf.ones(shape=(batch_len, 1), dtype=tf.int32)

        loss = self.loss_fn(labels, y_pred, feat_lens, label_lens)
        self.add_loss(loss)

        # At test time, just return the computed predictions
        return y_pred

def build_model(
    feat_dim,
    pad_index,
    num_layers,
    hidden_dim,
    bidirectional,
    projection_dim,
    num_tokens,
    rnn_type,
    initial_learning_rate,
    ):

    # Inputs to the model
    feats = layers.Input(
        name="feat", shape=(None, feat_dim), dtype=tf.float32
    )
    labels = layers.Input(name="label", shape=(None,), dtype=tf.float32)
    feat_lens = layers.Input(name="feat_len", shape=(1,), dtype=tf.int32)
    label_lens = layers.Input(name="label_len", shape=(1,), dtype=tf.int32)

    '''LeCunのパラメータ初期化方法の実行
    各重み(バイアス成分除く)を，平均0，標準偏差 1/sqrt(dim) の
    正規分布に基づく乱数で初期化(dim は入力次元数)
    model: Pytorchで定義したモデル
    '''
    initializer = tf.keras.initializers.LecunNormal()

    # First conv block
    x = layers.Masking(pad_index)(feats)
    for i in range(num_layers):
        if bidirectional:
            x = layers.Bidirectional(
                    layers.GRU(hidden_dim, return_sequences=True, kernel_initializer=initializer) if rnn_type == 'GRU' \
                        else layers.LTSM(hidden_dim, return_sequences=True, kernel_initializer=initializer))(x)
        else:
            x = layers.GRU(hidden_dim, return_sequences=True, kernel_initializer=initializer)(x) if rnn_type == 'GRU' \
                    else layers.LTSM(hidden_dim, return_sequences=True, kernel_initializer=initializer)(x)
        # Projection層もRNN層と同様に1層ずつ定義する
        # TODO sub sampling
        x = layers.Dense(projection_dim, kernel_initializer=initializer)(x)

    x = layers.Dense(num_tokens, name="softmax", activation=tf.nn.softmax, kernel_initializer=initializer)(x)

    # Add CTC layer for calculating CTC loss at each step
    output = CTCLayer(name="ctc_loss")(x, labels, feat_lens, label_lens)

    # Define the model
    model = keras.models.Model(
        inputs=[feats, labels, feat_lens, label_lens], outputs=output,
    )

    # Optimizer
    opt = keras.optimizers.Adadelta(learning_rate=initial_learning_rate, rho=0.95, epsilon=1e-8),
    opt = keras.optimizers.Adam()

    # Compile the model and return
    model.compile(
        optimizer=opt,
        # run_eagerly=True,
    )

    return model
