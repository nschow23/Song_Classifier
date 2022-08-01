#following code from https://towardsdatascience.com/simple-text-generation-d1c93f43f340

import pandas as pd
import tensorflow as tf
import tensorflow_text as tf_text
from tensorflow import keras


def main():
    ###import data and split into country and pop
    df_train = pd.read_csv("lyrics_plus.csv", header=None)
    df_train.columns = ['artist', 'lyrics', 'label']
    country = df_train[df_train.label == 1]
    X_country = country.lyrics
    y_country = country.label

    pop=df_train[df_train.label == 0]
    X_pop = pop.lyrics
    y_pop = pop.label

    max_words = 50000
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(X_country)
    sequences = tokenizer.texts_to_sequences(X_country)

    ## flatten
    text = [item for sublist in sequences for item in sublist]
    vocab_size = len(tokenizer.word_index)

    sentence_len = 10
    pred_len= 1
    train_len = sentence_len - pred_len
    seq = []
    ##sliding window to generate trian data
    for i in range(len(text)-sentence_len):
        seq.append(text[i:i+sentence_len])
    ##reverse dictionary to get words
    reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))

    trainX = []
    trainy = []
    for i in seq:
        trainX.append(i[:train_len])
        trainy.append(i[-1])

    model_2 = keras.Sequential([
        keras.Embedding(vocab_size+1, 50, input_length=train_len),
        keras.LSTM(100, return_sequences=True),
        keras.LSTM(100),
        keras.Dense(100, activation='relu'),
        keras.Dropout(0.1),
        keras.Dense(vocab_size, activation='softmax')
    ])

    ## train model with checkpoints
    model_2.compile(optimimzer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    filepath = "./model_2_weights.hdf5"
    checkpoint = keras.ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min'
                                        )
    callbacks_list = [checkpoint]
    history = model_2.fit(np.asarray(trainX),
                          pd.get_dummies(np.asarray(trainy)),
                          epochs=300,
                          batch_size=128,
                          callbacks=callbacks_list,
                          verbose=1)

    def gen(model, seq, max_len=20):
        ''' Generates a sequence given a string seq using specified model until the total sequence length
        reaches max_len'''
        # Tokenize the input string
        tokenized_sent = tokenizer.texts_to_sequences([seq])
        max_len = max_len + len(tokenized_sent[0])
        # If sentence is not as long as the desired sentence length, we need to 'pad sequence' so that
        # the array input shape is correct going into our LSTM. the `pad_sequences` function adds
        # zeroes to the left side of our sequence until it becomes 19 long, the number of input features.
        while len(tokenized_sent[0]) < max_len:
            padded_sentence = pad_sequences(tokenized_sent[-19:], maxlen=19)
            op = model.predict(np.asarray(padded_sentence).reshape(1, -1))
            tokenized_sent[0].append(op.argmax() + 1)

        return " ".join(map(lambda x: reverse_word_map[x], tokenized_sent[0]))

    seq = "Hey girl, get in my truck "

    output = gen(model_2, seq, max_len=10)
    print(output)
if __name__ == '__main__':
    main()