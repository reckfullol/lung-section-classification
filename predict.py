from keras.preprocessing import image
from keras.models import load_model
import numpy as np
import os
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

def predict(model, image_path):
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

if __name__ == '__main__':
	model = load_model('full_model.h5')
	print("Model load")

	rootdir = "D:\\PycharmProjects\\Image_Classifictaion\\test"

	for parent, dirnames, filenames in os.walk(rootdir):
		for filename in filenames:
			if '.jpg' in filename.lower():
				image_path = os.path.join(parent, filename)
				predict(model, image_path)
