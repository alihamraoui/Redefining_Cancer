import pickle as pickle
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.layers import SpatialDropout1D, Dropout, MaxPooling1D, Flatten
from tensorflow.keras.layers import Input, Dense, Conv1D
from tensorflow.keras import layers 
from tensorflow.keras import Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import to_categorical
from keras.callbacks import EarlyStopping


def read_file(pick):
    with open(pick,'rb') as infile :
        #with open("clean_text.pkl",'rb') as infile :
        data = pickle.load(infile)
    train_x, test_x, train_y, test_y = train_test_split(data[["Gene","Variation","TEXT"]],
                                                                        data["Class"], 
                                                                        random_state = 7,
                                                                        stratify=data["Class"],
                                                                        test_size=0.2)
    return train_x, test_x, train_y, test_y


def plot_classes(df):
    (df.value_counts(sort=False)).plot(kind='bar')
    plt.plot()
    plt.savefig('classes.png')


def tfidf_fun(train_x,test_x, length=6000):
    vocabulary_length = length
    tfidfvectorizer = TfidfVectorizer(analyzer='word',stop_words= 'english', max_features=vocabulary_length, use_idf=True)

    # convertir text to matrix
    tfidf_wm = tfidfvectorizer.fit_transform(train_x.TEXT).astype('float64')
    tfidf_wm_test = tfidfvectorizer.transform(test_x.TEXT).astype('float64')

    tfidf_tokens = tfidfvectorizer.get_feature_names()

    df_tfidfvect = pd.DataFrame(data = tfidf_wm.toarray(),index = train_x.Variation, columns = tfidf_tokens)
    df_tfidfvect_test = pd.DataFrame(data = tfidf_wm_test.toarray(),index = test_x.Variation, columns = tfidf_tokens)

    tf_len = len( tfidfvectorizer.get_feature_names())
    return df_tfidfvect, df_tfidfvect_test, tf_len


def build_model(tf_len):
    input = Input(shape=(tf_len, 1)) 
    x = SpatialDropout1D(0.3)(input)
    x = Conv1D(filters=128, kernel_size=5, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Conv1D(filters=128, kernel_size=5, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Conv1D(filters=128, kernel_size=5, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = Flatten()(x)
    out = Dense(9, activation='softmax')(x)
    model = Model(input, out)
    model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    return model


def main():
    train_x, test_x, train_y, test_y = read_file("complete_clean_bio.pkl")
    plot_classes(train_y,)

    #classes
    train_y = pd.DataFrame([int(y) - 1 for y in train_y])
    test_y = pd.DataFrame([int(y) - 1 for y in test_y])

    df_tfidfvect, df_tfidfvect_test, tf_len = tfidf_fun(train_x,test_x)
    
    #to Numpy array
    x_train = df_tfidfvect.to_numpy()
    x_test = df_tfidfvect_test.to_numpy()
    y_train = train_y.to_numpy()
    y_test = test_y.to_numpy()

    # Categorical
    y_train = to_categorical(y_train, num_classes=9, dtype ="uint8")
    y_test = to_categorical(y_test, num_classes=9, dtype ="uint8")

    #model
    model = build_model(tf_len)
    model.summary()

    #trinning
    history = model.fit(x_train, y_train, batch_size=140, epochs=20, verbose=1, validation_data = (x_test, y_test),  callbacks = EarlyStopping( monitor = "val_loss",patience=10)) 
    
    
if __name__ == '__main__':
    main()
