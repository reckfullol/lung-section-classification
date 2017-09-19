from keras.preprocessing import image
from keras.models import load_model
import numpy as np
import os


def predict(model, test_image):
	test_image = image.img_to_array(test_image)
	test_image = np.expand_dims(test_image, axis=0)
	test_image *= 1. / 255

	predict = model.predict(test_image)[0]
	pred_class = np.argmax(predict)
	label = ["high", "low", "medium", "normal"]

	print("class : {0}".format(label[pred_class]))
	print("prediction : {0}".format(predict[pred_class]))

if __name__ == '__main__':
	model = load_model('full_model.h5')
	print("Model load")

	rootdir = "D:\\PycharmProjects\\Image_Classifictaion\\validation"

	for parent, dirnames, filenames in os.walk(rootdir):
		for filename in filenames:
			if '.jpg' in filename.lower():
				image_path = os.path.join(parent, filename)
				img = image.load_img(image_path, target_size=(299, 299))
				print(image_path)
				predict(model, img)
