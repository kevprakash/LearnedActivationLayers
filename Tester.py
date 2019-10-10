import LearnedActivationLayer as LAL
import keras
import tensorflow as tf

def controlModel(inputShape, c, d, h, outputLength, act=keras.activations.relu):
    input = keras.layers.Input(shape=inputShape)
    x = input
    for _ in range(0, c[0]):
        for _ in range(0, c[1]):
            x = keras.layers.Conv2D(filters=h, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=act)(x)
        x = keras.layers.Conv2D(filters=h, kernel_size=(3, 3), strides=(2, 2), padding='same', activation=act)(x)
    x = keras.layers.Flatten()(x)
    for _ in range(0, d):
        x = keras.layers.Dense(units=h, activation=act)(x)
    output = keras.layers.Dense(units=outputLength, activation=keras.activations.softmax)(x)

    model = keras.Model(inputs=[input], outputs=[output])
    model.compile(optimizer=keras.optimizers.RMSprop(lr=0.0002), loss=keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])

    return model


def tuningModel(inputShape, c, d, h, outputLength, act=keras.activations.relu):
    input = keras.layers.Input(shape=inputShape)
    x = input
    for _ in range(0, c[0]):
        for _ in range(0, c[1]):
            x = keras.layers.Conv2D(filters=h, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(x)
            x = LAL.TuningActivationLayer(activationFunction=act)(x)
        x = keras.layers.Conv2D(filters=h, kernel_size=(3, 3), strides=(2, 2), padding='same', activation=None)(x)
        x = LAL.TuningActivationLayer(activationFunction=act)(x)
    x = keras.layers.Flatten()(x)
    for _ in range(0, d):
        x = keras.layers.Dense(units=h, activation=None)(x)
        x = LAL.TuningActivationLayer(activationFunction=act)(x)
    x = keras.layers.Dense(units=outputLength, activation=None)(x)
    x = LAL.TuningActivationLayer(activationFunction=keras.activations.softmax)(x)

    model = keras.Model(inputs=[input], outputs=[x])
    model.compile(optimizer=keras.optimizers.RMSprop(lr=0.0002), loss=keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])

    return model


def learningModel(inputShape, c, d, h, outputLength, N, activationLayer):
    input = keras.layers.Input(shape=inputShape)
    x = input
    for _ in range(0, c[0]):
        for _ in range(0, c[1]):
            x = keras.layers.Conv2D(filters=h, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(x)
            x = activationLayer(N)(x)
        x = keras.layers.Conv2D(filters=h, kernel_size=(3, 3), strides=(2, 2), padding='same', activation=None)(x)
        x = activationLayer(N)(x)
    x = keras.layers.Flatten()(x)
    for _ in range(0, d):
        x = keras.layers.Dense(units=h, activation=None)(x)
        x = activationLayer(N)(x)
    x = keras.layers.Dense(units=outputLength, activation=keras.activations.softmax)(x)

    model = keras.Model(inputs=[input], outputs=[x])
    model.compile(optimizer=keras.optimizers.RMSprop(lr=0.0002), loss=keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])

    return model


def modelTester(c, d, h, N, learnedLayer, act=keras.activations.relu):
    inShape = (32, 32, 3)
    dataset = keras.datasets.cifar10
    (train_images, train_labels), (test_images, test_labels) = dataset.load_data()

    cm = controlModel(inShape, c, d, h, 10, act)
    tm = tuningModel(inShape, c, d, h, 10, act)
    lm = learningModel(inShape, c, d, h, 10, N, learnedLayer)

    print("Learning Model")
    lm.fit(train_images, train_labels, epochs=5)
    print("Tuning Model")
    tm.fit(train_images, train_labels, epochs=5)
    print("Control Model")
    cm.fit(train_images, train_labels, epochs=5)

    results = []
    results.append(lm.evaluate(test_images, test_labels))
    results.append(tm.evaluate(test_images, test_labels))
    results.append(cm.evaluate(test_images, test_labels))

    print(results)

    return results


test_results = []
for _c in range(1, 4):
    res_c = []
    for _c2 in range(1, 4):
        res_c2 = []
        for _d in range(1, 4):
            res_d = []
            for _h in range(5, 8):
                res_d.append(modelTester((_c, _c2), _d, 2**_h, 6, LAL.LearnedExponentialActivationLayer, keras.activations.relu))
            res_c2.append(res_d)
        res_c.append(res_c2)
    test_results.append(res_c)


for _c in range(0, 3):
    for _c2 in range(0, 3):
        for _d in range(0, 3):
            for _h in range(0, 3):
                print("Learned Activation (" + str(_c) + ", " + str(_c2) + ", " + str(_d) + ", " + str(2**(_h + 5)) + "): Loss = " + test_results[_c][_c2][_d][_h][0][0] + " Accuracy = " + test_results[_c][_c2][_d][_h][0][1])
                print("Tuning  Activation (" + str(_c) + ", " + str(_c2) + ", " + str(_d) + ", " + str(2**(_h + 5)) + "): Loss = " + test_results[_c][_c2][_d][_h][1][0] + " Accuracy = " + test_results[_c][_c2][_d][_h][1][1])
                print("Control Activation (" + str(_c) + ", " + str(_c2) + ", " + str(_d) + ", " + str(2**(_h + 5)) + "): Loss = " + test_results[_c][_c2][_d][_h][2][0] + " Accuracy = " + test_results[_c][_c2][_d][_h][2][1])