import tensorflow as tf
import numpy as np
import pennylane as qml

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

# see also: https://www.tensorflow.org/tutorials/text/transformer


USE_GPU = bool(os.environ.get('USE_GPU', False))
#USE_GPU = True

def get_angles(pos, i, embed_dim):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(embed_dim))
  return pos * angle_rates


def positional_encoding(position, embed_dim):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(embed_dim)[np.newaxis, :],
                          embed_dim)

  # apply sin to even indices in the array; 2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

  # apply cos to odd indices in the array; 2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

  pos_encoding = angle_rads[np.newaxis, ...]

  return tf.cast(pos_encoding, dtype=tf.float32)


def create_padding_mask(seq):
  seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

  # add extra dimensions to add the padding
  # to the attention logits.
  return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
  mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
  return mask  # (seq_len, seq_len)


def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)  
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)
    return output, attention_weights


class MultiHeadAttentionBase(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttentionBase, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim

        assert embed_dim % self.num_heads == 0

        self.depth = embed_dim // self.num_heads

        self.wq = None
        self.wk = None
        self.wv = None
        self.dense = None

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
         Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def apply_dense_layers(self, v, k, q):
        raise NotImplementedError("Base class does not implement apply_dense_layers() function")

    def apply_combine_heads(self, x):
        raise NotImplementedError("Base class does not implement apply_combine_heads() function")

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
#         print("v", tf.shape(v))
#         print("k", tf.shape(k))
#         print("q", tf.shape(q))
        v, k, q = self.apply_dense_layers(v, k, q)
#         print("v", tf.shape(v))
#         print("k", tf.shape(k))
#         print("q", tf.shape(q))
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.embed_dim))  # (batch_size, seq_len_q, embed_dim)

        output = self.apply_combine_heads(concat_attention) # (batch_size, seq_len_q, embed_dim)
        return output, attention_weights


class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttentionClassical, self).__init__(embed_dim, num_heads)
        self.wq = tf.keras.layers.Dense(embed_dim)
        self.wk = tf.keras.layers.Dense(embed_dim)
        self.wv = tf.keras.layers.Dense(embed_dim)
        self.dense = tf.keras.layers.Dense(embed_dim)
    
    def apply_dense_layers(self, v, k, q):
#         print("q type", type(q))
#         print(q.shape)
#         print(tf.shape(q))

        q = self.wq(q)  # (batch_size, seq_len, embed_dim)
        k = self.wk(k)  # (batch_size, seq_len, embed_dim)
        v = self.wv(v)  # (batch_size, seq_len, embed_dim)
        return v, k, q
    
    def apply_combine_heads(self, x):
        return self.dense(x)  # (batch_size, seq_len_q, embed_dim)


class MultiHeadAttentionQuantum(MultiHeadAttentionBase):
    def __init__(self, 
                 embed_dim, num_heads, 
                 n_qubits, n_qlayers=1, q_device='default.qubit',ansatz_id=1):
        super(MultiHeadAttentionQuantum, self).__init__(embed_dim, num_heads)
        # todo: add intermediate layer to "dress" quantum circuit
        assert n_qubits == embed_dim, f"Number of qubits ({n_qubits}) does not match embedding dim ({embed_dim})"
        
        self.n_qubits = n_qubits
        self.n_qlayers = n_qlayers
        self.q_device = q_device
        self.ansatz_id = ansatz_id
        
        if 'qulacs' in q_device:
            print(f"Quantum device: Qulacs: {q_device}")
            if USE_GPU is True:
                print("Qulacs will use the GPU")
            self.dev = qml.device(q_device, wires=n_qubits, gpu=USE_GPU)
        elif 'braket' in q_device:
            print(f"Quantum device: Amazon Braket: {q_device}")
            self.dev = qml.device(q_device, wires=n_qubits, parallel=True)
        else:
            print(f"Quantum device: {q_device}")
            self.dev = qml.device(q_device, wires=n_qubits)
        
        weight_shapes = {"weights": (n_qlayers, n_qubits)}
        print(f"weight_shapes = (n_qlayers, n_qubits) = ({n_qlayers}, {n_qubits})")
        
        if ansatz_id==1:
            def _circuit(inputs, weights):
                qml.templates.AngleEmbedding(inputs, wires=range(self.n_qubits))
                qml.templates.BasicEntanglerLayers(weights, wires=range(n_qubits))
                return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]
        elif ansatz_id==2:
            def _circuit(inputs, weights):
                for i in range(len(inputs)):
                    qml.Hadamard(wires=i)
                inputs_arctan1 = np.arctan(inputs)
                inputs_arctan2 = np.arctan(inputs**2)
                qml.templates.AngleEmbedding(inputs_arctan1, wires=range(n_qubits),rotation='Y')
                qml.templates.AngleEmbedding(inputs_arctan2, wires=range(n_qubits),rotation='Z')
                for w in range(weights.shape[0]):
                  qml.broadcast(qml.CNOT,wires=range(n_qubits),pattern="ring")
                  for i in range(len(inputs)):
                    qml.CNOT(wires=[i, (i+2)%len(inputs)])
                  qml.templates.AngleEmbedding(weights[w], wires=range(n_qubits),rotation='Y')
                 
                return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]
                
        self.qlayer = qml.QNode(_circuit, self.dev, interface="tf")
        
        self.wq = qml.qnn.KerasLayer(self.qlayer, weight_shapes, output_dim=n_qubits)
        self.wk = qml.qnn.KerasLayer(self.qlayer, weight_shapes, output_dim=n_qubits)
        self.wv = qml.qnn.KerasLayer(self.qlayer, weight_shapes, output_dim=n_qubits)
        self.dense = qml.qnn.KerasLayer(self.qlayer, weight_shapes, output_dim=n_qubits)

    def apply_dense_layers(self, v, k, q):
#         print("q type", type(q))
#         print(q.shape)
#         print(q)
#         print(tf.shape(q))
#         seq_len = q.shape[1]
#         seq_len = tf.convert_to_tensor(seq_len, dtype=tf.int32)
        try :
            batch_size, seq_len, _ = tf.shape(q)
#             print(type(seq_len))
#             print(seq_len)
#             print(tf.convert_to_tensor(6, dtype=tf.int32))
        except:
            seq_len = q.shape[1]
#         batch_size, seq_len, _ = q.shape

#         q = tf.map_fn(lambda t: self.wq(q[:, t, :]), tf.range(seq_len)) 
#         k = tf.map_fn(lambda t: self.wk(k[:, t, :]), tf.range(seq_len)) 
#         v = tf.map_fn(lambda t: self.wv(v[:, t, :]), tf.range(seq_len)) 
        q = [self.wq(q[:, t, :]) for t in range(seq_len)]  # (seq_len, batch_size, embed_dim)
        k = [self.wk(k[:, t, :]) for t in range(seq_len)]  # (seq_len, batch_size, embed_dim)
        v = [self.wv(v[:, t, :]) for t in range(seq_len)]  # (seq_len, batch_size, embed_dim)

        q = tf.convert_to_tensor(q)
        k = tf.convert_to_tensor(k)
        v = tf.convert_to_tensor(v)

        q = tf.transpose(q, perm=[1, 0, 2])  # (batch_size, seq_len, embed_dim)
        k = tf.transpose(k, perm=[1, 0, 2])  # (batch_size, seq_len, embed_dim)
        v = tf.transpose(v, perm=[1, 0, 2])  # (batch_size, seq_len, embed_dim)
        
#         print("q type-post", type(q))
#         print(q.shape)
#         print(tf.shape(q))
        return v, k, q
    
    def apply_combine_heads(self, x):
        _, seq_len, _ = tf.shape(x)
        x = [self.dense(x[:, t, :]) for t in range(seq_len)]  # (seq_len, batch_size, embed_dim)
        x = tf.convert_to_tensor(x)
        x = tf.transpose(x, perm=[1,0,2])
        return x


def point_wise_feed_forward_network_classical(embed_dim, dff):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(embed_dim)  # (batch_size, seq_len, embed_dim)
  ])


def point_wise_feed_forward_network_quantum(embed_dim, dff, n_qubits_ffn, n_qlayers=1, q_device='default.qubit'):
    return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(embed_dim)  # (batch_size, seq_len, embed_dim)
  ])


class TransformerBlockBase(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, dff, dropout_rate=0.1):
        super(TransformerBlockBase, self).__init__()
        self.mha = None
        self.ffn = None

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, embed_dim)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, embed_dim)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, embed_dim)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, embed_dim)

        return out2


class TransformerBlockClassical(TransformerBlockBase):
    def __init__(self, embed_dim, num_heads, dff, dropout_rate=0.1):
        super(TransformerBlockClassical, self).__init__(embed_dim, num_heads, dff, dropout_rate)
        self.mha = MultiHeadAttentionClassical(embed_dim, num_heads)
        self.ffn = point_wise_feed_forward_network_classical(embed_dim, dff)


class TransformerBlockQuantum(TransformerBlockBase):
    def __init__(self, 
                 embed_dim, num_heads, dff, dropout_rate=0.1,
                 n_qubits_transformer: int = 0,
                 n_qubits_ffn: int = 0,
                 n_qlayers: int = 1,
                 q_device='default.qubit',
                 ansatz_id=1):
        super(TransformerBlockQuantum, self).__init__(embed_dim, num_heads, dff, dropout_rate)
        self.mha = MultiHeadAttentionQuantum(embed_dim, num_heads, n_qubits_transformer, n_qlayers, q_device, ansatz_id)
        self.ffn = point_wise_feed_forward_network_quantum(embed_dim, dff, n_qubits_ffn, n_qlayers, q_device)


class EncoderLayerBase(tf.keras.layers.Layer):
    def __init__(self, 
                num_layers, 
                embed_dim, 
                num_heads, 
                dff, 
                vocab_size,
                maximum_position_encoding, 
                dropout_rate=0.1):
        super(EncoderLayerBase, self).__init__()

        self.embed_dim = embed_dim
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_dim)
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.embed_dim)
        self.enc_layers = None
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training, mask=None):

        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, embed_dim)
        x *= tf.math.sqrt(tf.cast(self.embed_dim, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, embed_dim)


class EncoderLayerClassical(EncoderLayerBase):
    def __init__(self, 
                num_layers, 
                embed_dim, 
                num_heads, 
                dff, 
                vocab_size,
                maximum_position_encoding, 
                dropout_rate=0.1):
        super(EncoderLayerClassical, self).__init__(num_layers, embed_dim, num_heads, dff, vocab_size, maximum_position_encoding, dropout_rate)
        
        self.enc_layers = [TransformerBlockClassical(embed_dim, num_heads, dff, dropout_rate) 
                        for _ in range(num_layers)]


class EncoderLayerQuantum(EncoderLayerBase):
    def __init__(self, 
                num_layers, 
                embed_dim, 
                num_heads, 
                dff, 
                vocab_size,
                maximum_position_encoding, 
                dropout_rate=0.1,
                n_qubits_transformer: int = 0,
                n_qubits_ffn: int = 0,
                n_qlayers: int = 1,
                q_device="device.qubit",
                ansatz_id=1):
        super(EncoderLayerQuantum, self).__init__(num_layers, embed_dim, num_heads, dff, vocab_size, maximum_position_encoding, dropout_rate)
        self.enc_layers = [TransformerBlockQuantum(embed_dim, num_heads, dff, dropout_rate, 
                                                   n_qubits_transformer, n_qubits_ffn, n_qlayers, q_device, ansatz_id)
                            for _ in range(num_layers)]


class TextClassifierTF(tf.keras.Model):
    def __init__(self, 
                num_layers, 
                embed_dim, 
                num_heads, 
                dff, 
                vocab_size, 
                num_classes, 
                maximum_position_encoding: int=10000, 
                dropout_rate=0.1,
                n_qubits_transformer: int = 0,
                n_qubits_ffn: int = 0,
                n_qlayers: int = 1,
                q_device="device.qubit",
                ansatz_id=1):
        super(TextClassifierTF, self).__init__()

        if n_qubits_transformer == 0 and n_qubits_ffn == 0:
            self.encoder = EncoderLayerClassical(num_layers, embed_dim, num_heads, dff, 
                            vocab_size, maximum_position_encoding, dropout_rate)
        else:
            self.encoder = EncoderLayerQuantum(num_layers, embed_dim, num_heads, dff, 
                            vocab_size, maximum_position_encoding, dropout_rate,
                            n_qubits_transformer, n_qubits_ffn, n_qlayers, q_device, ansatz_id)
        
        if num_classes < 2:
            raise RuntimeError("Number of classes must be at least 2")
        elif num_classes == 2:
            self.final_layer = tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid)
        else:
            self.final_layer = tf.keras.layers.Dense(num_classes, activation=tf.keras.activations.softmax)
    
    def call(self, x, training):
        encoded_output = self.encoder(x, training)  # (batch_size, inp_seq_len, embed_dim)
        pooled_output = encoded_output[:,0,:]
        final_output = self.final_layer(pooled_output)  # (batch_size, tar_seq_len, num_classes)

        return final_output