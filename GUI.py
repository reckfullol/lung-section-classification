import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog
from PyQt5.QtGui import QIcon
from keras.preprocessing import image
from keras.models import load_model
import numpy as np
import os
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

model = load_model('full_model.h5')


def predict(image_path):
	test_image = image.load_img(image_path, target_size=(299, 299))
	test_image = image.img_to_array(test_image)
	test_image = np.expand_dims(test_image, axis=0)
	test_image *= 1. / 255

	predict = model.predict(test_image)[0]
	pred_class = np.argmax(predict)
	label = ["high", "low", "medium", "normal"]

	print('----------------------')
	print(image_path)
	print("Class : {0}".format(label[pred_class]))
	print("Confidence rate : {0}%".format(predict[pred_class] * 100))
	print('\n\n\n')

	img = Image.open(image_path)
	draw = ImageDraw.Draw(img)
	# font = ImageFont.truetype(<font-file>, <font-size>)
	font = ImageFont.truetype("arial.ttf", 16)
	# draw.text((x, y),"Sample Text",(r,g,b))
	draw.text((0, 0), "Class : {0}\nConfidence rate : {1}%".format(label[pred_class], predict[pred_class] * 100), (255, 255, 255), font=font)
	img.show()

class App(QWidget):
	def __init__(self):
		super().__init__()
		self.title = 'PyQt5'
		self.left = 10
		self.top = 10
		self.width = 320
		self.height = 200
		self.initUI()

	def initUI(self):
		self.setWindowTitle(self.title)
		self.setGeometry(self.left, self.top, self.width, self.height)

		button = QPushButton('select', self)
		button.setToolTip('This is an example button')
		button.move(100, 70)
		button.clicked.connect(self.on_click)

		self.show()

	@pyqtSlot()
	def on_click(self):
		options = QFileDialog.Options()
		options |= QFileDialog.DontUseNativeDialog
		fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
				"All Files (*);;Python Files (*.py)", options=options)
		if fileName:
			predict(fileName)


if __name__ == '__main__':
	app = QApplication(sys.argv)
	ex = App()
	sys.exit(app.exec_())