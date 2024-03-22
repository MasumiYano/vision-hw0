from typing import List, Dict
from src import classifier, data


def make_model(layers: List[Dict]) -> Dict:
    model = {
        'layers': layers,
        'n': len(layers)
    }
    return model


def softmax_model(inputs: int, outputs: int) -> Dict:
    softmax_layer = classifier.make_layer(inputs, outputs, 'SOFTMAX')
    return make_model([softmax_layer])


def relu_model(inputs: int, outputs: int) -> Dict:
    l = [
        classifier.make_layer(inputs, 32, 'RELU'),
        classifier.make_layer(32, outputs, 'SOFTMAX')
    ]
    return make_model(l)


def neural_net(inputs, outputs):
    print(inputs)
    l = [
        classifier.make_layer(inputs, 32, 'LOGISTIC'),
        classifier.make_layer(32, outputs, 'SOFTMAX')
    ]
    return make_model(l)


print('Loading data...')
train = data.load_classification_data("mnist.train", "mnist.labels", True)
test = data.load_classification_data("mnist.test", "mnist.labels", True)
print("Done")

print("Training model...")
batch = 128
iters = 1000
rate = 0.01
momentum = 0.9
decay = 0.1

model = softmax_model(train['X'].shape[1], train['y'].shape[1])
classifier.train_model(model, train, batch, iters, rate, momentum, decay)
print("Done")

print("evaluating model...")
print(f"Training accuracy: {classifier.accuracy_model(model, train)}")
print(f'Test accuracy: {classifier.accuracy_model(model, test)}')
