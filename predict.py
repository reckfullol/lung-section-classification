from keras.preprocessing import image
from keras.models import load_model
from keras.applications.inception_v3 import preprocess_input
import numpy as np
from PIL import Image
import os

def predict(model, img):
	img = img.resize((299, 299))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	preds = model.predict(x)[0]
	print(preds)
	pred_class = np.argmax(preds)
	print(pred_class)


if __name__ == '__main__':
	model = load_model('full_model.h5')
	print("Model load")

	rootdir = "./train/high"

	for parent, dirnames, filenames in os.walk(rootdir):
		for filename in filenames:
			if '.jpg' in filename.lower():
				image_path = os.path.join(parent, filename)
				img = Image.open(image_path)
				print(image_path)
				predict(model, img)
#model = load_model('full_model.h5')
#img_path = './test/high/high_test.jpg'
#img = Image.open(img_path)
#preds = predict(model, img)
