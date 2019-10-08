import LearnedActivationLayer as LAL
import keras
import tensorflow as tf

def LALTester(c=1, d=2, act=keras.activations.relu):
    dataset = keras.datasets.cifar10
    input_shape = (32, 32, 3)
    reshape_shape = (32, 32, 3)
    outputLength = 10
    h = 32
    drop = 0.0

    (train_images, train_labels), (test_images, test_labels) = dataset.load_data()

    train_images = train_images / 255.0

    test_images = test_images / 255.0

    input = keras.layers.Input(shape=input_shape)
    reshape = keras.layers.Reshape(target_shape=reshape_shape)(input)
    x = keras.layers.Conv2D(filters=h, kernel_size=(3, 3), strides=(2, 2), padding='same')(reshape)
    x = LAL.LearnedActivationLayer(activationFunction=act)(x)
    x = keras.layers.Dropout(rate=drop)(x)
    for _ in range(0, c):
        x = keras.layers.Conv2D(filters=h, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
        x = LAL.LearnedActivationLayer(activationFunction=act)(x)
        x = keras.layers.Dropout(rate=drop)(x)
    x = keras.layers.Flatten()(x)
    for _ in range(0, d):
        x = keras.layers.Dense(units=h, activation=None)(x)
        x = LAL.LearnedActivationLayer(activationFunction=act)(x)
    x = keras.layers.Dense(units=outputLength, activation=None)(x)
    x = LAL.LearnedActivationLayer(activationFunction=keras.activations.softmax)(x)

    model = keras.Model(inputs=[input], outputs=[x])
    model.compile(optimizer=keras.optimizers.RMSprop(), loss=keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])

    #########################################

    input2 = keras.layers.Input(shape=input_shape)
    reshape2 = keras.layers.Reshape(target_shape=reshape_shape)(input2)
    x2 = keras.layers.Conv2D(filters=h, kernel_size=(3, 3), strides=(2, 2), padding='same')(reshape2)
    x2 = LAL.LearnedFourierActivationLayer(paramCount=4)(x2)
    x2 = keras.layers.Dropout(rate=drop)(x2)
    for _ in range(0, c):
        x2 = keras.layers.Conv2D(filters=h, kernel_size=(3, 3), strides=(2, 2), padding='same')(x2)
        x2 = LAL.LearnedFourierActivationLayer(paramCount=4)(x2)
        x2 = keras.layers.Dropout(rate=drop)(x2)
    x2 = keras.layers.Flatten()(x2)
    for _ in range(0, d):
        x2 = keras.layers.Dense(units=h, activation=None)(x2)
        x2 = LAL.LearnedFourierActivationLayer(paramCount=4)(x2)
    x2 = keras.layers.Dense(units=outputLength, activation=keras.activations.softmax)(x2)

    model2 = keras.Model(inputs=[input2], outputs=[x2])
    model2.compile(optimizer=keras.optimizers.RMSprop(), loss=keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])

    #########################################

    input3 = keras.layers.Input(shape=reshape_shape)
    reshape3 = keras.layers.Reshape(target_shape=reshape_shape)(input3)
    x3 = keras.layers.Conv2D(filters=h, kernel_size=(3, 3), strides=(2, 2), padding='same', activation=act)(reshape3)
    x3 = keras.layers.Dropout(rate=drop)(x3)
    for _ in range(0, c):
        x3 = keras.layers.Conv2D(filters=h, kernel_size=(3, 3), strides=(2, 2), padding='same', activation=act)(x3)
        x3 = keras.layers.Dropout(rate=drop)(x3)
    x3 = keras.layers.Flatten()(x3)
    for _ in range(0, d):
        x3 = keras.layers.Dense(units=h, activation=act)(x3)
    x3 = keras.layers.Dense(units=outputLength, activation=keras.activations.softmax)(x3)

    model3 = keras.Model(inputs=[input3], outputs=[x3])
    model3.compile(optimizer=keras.optimizers.RMSprop(), loss=keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])

    #print(model.predict(train_images))

    model.fit(train_images, train_labels, epochs=5, verbose=True)
    model2.fit(train_images, train_labels, epochs=5, verbose=True)
    model3.fit(train_images, train_labels, epochs=5, verbose=True)

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=True)
    test_loss2, test_acc2 = model2.evaluate(test_images, test_labels, verbose=True)
    test_loss3, test_acc3 = model3.evaluate(test_images, test_labels, verbose=True)

    print('(' + str(c) + ', ' + str(d) + ') Test accuracy:', test_acc)
    print('(' + str(c) + ', ' + str(d) + ') Test accuracy 2:', test_acc2)
    print('(' + str(c) + ', ' + str(d) + ') Test accuracy 3:', test_acc3)

    return test_acc, test_acc2, test_acc3, '(' + str(c) + ', ' + str(d) + ') Test accuracy:' + str(test_acc), '(' + str(c) + ', ' + str(d) + ') Test accuracy 2:' + str(test_acc2), '(' + str(c) + ', ' + str(d) + ') Test accuracy 3:' + str(test_acc3)


testResults = []
for i in range(0, 4):
    arr = []
    for j in range(0, 4):
        arr.append(LALTester(c=i, d=j, act=keras.activations.tanh))
    testResults.append(arr)

print("Final Results:")

for i in range(len(testResults)):
    for j in range(len(testResults[i])):
        print(testResults[i][j][3])
        print(testResults[i][j][4])
        print(testResults[i][j][5])
        print()
