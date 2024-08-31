# dog-breed-classification-model
## A supervised learning classification model to label dog breeds using a convolutional neural network.
Can be run in a notebook with python or in an environment that supports python.
### About
This project was carried out to develop my understanding of supervised learning models and practice using convolutional neural networks alongside key machine learning libraries including TensorFlow, Keras, Pandas, and NumPy. The Stanford Dogs Dataset from [Kaggle](https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset), containing 120 different breeds was the dataset used in this project. The ResNet-50 CNN was used due to its effectiveness of classifying images, along with ReLU and Softmax activation functions to ensure that the outputs of each neural layer are in a suitable format.

CNN model:
```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ input_layer_21 (InputLayer)     │ (None, 224, 224, 3)    │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ resnet50 (Functional)           │ (None, 7, 7, 2048)     │    23,587,712 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ global_average_pooling2d_1      │ (None, 2048)           │             0 │
│ (GlobalAveragePooling2D)        │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_2 (Dense)                 │ (None, 256)            │       524,544 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_1 (Dropout)             │ (None, 256)            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_3 (Dense)                 │ (None, 120)            │        30,840 │

Total params: 24,143,096 (92.10 MB)
 Trainable params: 555,384 (2.12 MB)
 Non-trainable params: 23,587,712 (89.98 MB)
```
### Evaluation
```
Epoch 1/10
386/386 ━━━━━━━━━━━━━━━━━━━━ 3405s 9s/step - accuracy: 0.2748 - loss: 2.6979 - val_accuracy: 0.5224 - val_loss: 1.8128
Epoch 2/10
386/386 ━━━━━━━━━━━━━━━━━━━━ 1701s 4s/step - accuracy: 0.3633 - loss: 2.2847 - val_accuracy: 0.5447 - val_loss: 1.6527
Epoch 3/10
386/386 ━━━━━━━━━━━━━━━━━━━━ 1701s 4s/step - accuracy: 0.4040 - loss: 2.0997 - val_accuracy: 0.5617 - val_loss: 1.5546
Epoch 4/10
386/386 ━━━━━━━━━━━━━━━━━━━━ 1759s 5s/step - accuracy: 0.4213 - loss: 2.0049 - val_accuracy: 0.5797 - val_loss: 1.4927
Epoch 5/10
386/386 ━━━━━━━━━━━━━━━━━━━━ 1813s 5s/step - accuracy: 0.4410 - loss: 1.8871 - val_accuracy: 0.5828 - val_loss: 1.4598
Epoch 6/10
386/386 ━━━━━━━━━━━━━━━━━━━━ 2051s 5s/step - accuracy: 0.4540 - loss: 1.8428 - val_accuracy: 0.5790 - val_loss: 1.4589
Epoch 7/10
386/386 ━━━━━━━━━━━━━━━━━━━━ 3362s 9s/step - accuracy: 0.4712 - loss: 1.7706 - val_accuracy: 0.5724 - val_loss: 1.4775
Epoch 8/10
386/386 ━━━━━━━━━━━━━━━━━━━━ 2049s 5s/step - accuracy: 0.4892 - loss: 1.6999 - val_accuracy: 0.5804 - val_loss: 1.4518
Epoch 9/10
386/386 ━━━━━━━━━━━━━━━━━━━━ 2066s 5s/step - accuracy: 0.4948 - loss: 1.6725 - val_accuracy: 0.5882 - val_loss: 1.4257
Epoch 10/10
386/386 ━━━━━━━━━━━━━━━━━━━━ 2078s 5s/step - accuracy: 0.4972 - loss: 1.6175 - val_accuracy: 0.5984 - val_loss: 1.4130
```
Due to the large quantity of images (20,580) contained in the dataset, the training and evaluation stages took an extremely long time, therefore only ten epochs were able to be carried out, reducing the potential of obtaining an accurate model. As a result, by the. tenth epoch, there was a training accuracy of 49.72% and validation accuracy of 59.84%, however the consistent positive increase in accuracy suggests that more epochs may result in higher accuracies. To increase the model's accuracy, the hyperparameters of the CNN could be altered to find the most suitable values, and the neural network could be further customised by changing the number and type of layers within it.
