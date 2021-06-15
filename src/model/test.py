import keras.applications.resnet

def test_passing_class():
    print("test")
    hello(keras.applications.resnet)

def hello(model):
    print("model")
    print(model)
    print(str(model.__name__))
    print(model.__class__)
    resnet = model(include_top=True,
              weights='imagenet',
              input_tensor=None,
              input_shape=None,
              pooling=None,
              classes=1000)
    print(resnet)


if __name__ == '__main__':
    test_passing_class()