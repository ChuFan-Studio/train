import tensorflow as tf
from PIL import Image
import numpy as np
from train import CNN
from flask import Flask

app = Flask(__name__)
class Predict(object):
    def init(self):
        latest = tf.train.latest_checkpoint('./ckpt')
        self.cnn = CNN()
        # 恢复网络权重
        self.cnn.model.load_weights(latest)

    def predict(self, image_path):
        # 以黑白方式读取图片
        img = Image.open(image_path).convert('L')
        flatten_img = np.reshape(img, (28, 28, 1))
        x = np.array([1 - flatten_img])

        # API refer: https://keras.io/models/model/
        y = self.cnn.model.predict(x)

        # 因为x只传入了一张图片，取y[0]即可
        # np.argmax()取得最大值的下标，即代表的数字
        print(image_path)
        print(y[0])
        print('        -> Predict digit', np.argmax(y[0]))
        return y[0]

# global pre
pre = Predict() 

@app.before_first_request
def initMode():
    print('this is a init of cnn')
    pre.init()
    pass


@app.route('/')
def hello_world():
    ret = pre.predict('./0.png')
    s="Return:["
    for v in ret:
        s=s+str(v)+","
    return s[0:len(s)-1]+"]\n"

if __name__ == '__main__':
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        for k in range(len(physical_devices)):
            tf.config.experimental.set_memory_growth(physical_devices[k], True)
            print('memory growth:', tf.config.experimental.get_memory_growth(physical_devices[k]))
    else:
        print("Not enough GPU hardware devices available")
    app.run(debug=1)
