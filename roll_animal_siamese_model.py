import time
import numpy as np
from PIL import Image
import onnxruntime as ort


class CustomTransform:
    def __init__(self, size=(48, 48)):
        self.size = size

    def __call__(self, image):
        # 调整图像大小
        image = image.resize(self.size)

        # 将图像转换为 NumPy 数组
        image_array = np.array(image)

        # 将数值范围缩放到 [0, 1]
        image_array = image_array / 255.0

        # 增加批次维度
        input_array = image_array[np.newaxis, ...]

        # 如果需要添加通道维度（假设图像是 RGB 彩色图像），可以使用以下代码
        input_array = input_array.transpose(0, 3, 1, 2)

        return input_array


class RollAnimalSiameseModel:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(RollAnimalSiameseModel, cls).__new__(cls)
            cls._instance.__initialized = False
        return cls._instance

    def __init__(self, onnx_model_path):
        if self.__initialized:
            return
        self.onnx_model_path = onnx_model_path
        self.model = None
        self.transform = None
        self.session = None
        self.__initialized = True

    def _initialize_model(self):
        self.session = ort.InferenceSession(self.onnx_model_path)
        self.transform = CustomTransform()

    def predict_image(self, idx, input_finger_tensor, sub_image_tensor):
        input_finger_name = self.session.get_inputs()[1].name
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name
        prediction = self.session.run(
            [output_name],
            {
                input_name: sub_image_tensor.astype(np.float32),
                input_finger_name: input_finger_tensor.astype(np.float32),
            },
        )
        return prediction

    def predict(self, image):
        if self.session is None or self.transform is None:
            self._initialize_model()
        image_size = (200, 200)
        width, height = image.size
        rows = height // image_size[1]
        cols = width // image_size[0]
        finger_image = image.crop((0, image_size[1], image_size[0], 2 * image_size[1]))
        input_finger_tensor = self.transform(finger_image)
        tasks = []
        for i in range(cols):
            x_min, y_min, x_max, y_max = (
                i * image_size[0],
                0,
                (i + 1) * image_size[0],
                image_size[1],
            )
            sub_image = image.crop((x_min, y_min, x_max, y_max))
            sub_image_tensor = self.transform(sub_image)
            tasks.append(self.predict_image(i, input_finger_tensor, sub_image_tensor))

        max_index = np.argmax(tasks)
        return max_index


# def main(image_file, onnx_model_path):
#     model = RollAnimalSiameseModel(onnx_model_path)
#     image = Image.open(image_file)
#     start = time.time()
#     index = model.predict(image)
#     end = time.time()
#     print("预测耗时:", end - start)
#     print("预测结果:", index)
#
#
# image_file = '184655b7f772919c1_0_answer4.jpg'
# onnx_model_path = 'model.onnx'
# main(image_file, onnx_model_path)