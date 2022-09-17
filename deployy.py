import os
from flask import Flask,request
from flask import render_template
import io
from PIL import Image as PImage
#import fastbook
from fastai.vision import *
from fastai.callbacks.hooks import *
from fastai.vision.learner import *
import cv2
# import torch
# import pickle
# import torchvision.transforms as transforms
# import torch.utils.data
#from fastai import load_learner
#import fastai.callbacks.all

app = Flask(__name__)

upload_folder = "./statics"
model_path="./models/"
learnn = load_learner(os.path.join(model_path))

def make_prediction(image_file):
	model = learnn.model.eval()
	pred = learnn.predict(Image(image_file))
	print(pred)
	cls = int(pred[1])
	classes = ['COVID','NORMAL','PNEUMONIA']
	return str(classes[cls]) 
	# with open(model_path, 'rb') as file:
	# 	pickle_model = pickle.load(file)
	# predict = pickle_model.predict(Xtest)

def visualize(img):
	b,_ = learnn.data.one_item(img)
	img = Image(learnn.data.denorm(b)[0])
	with hook_output(model[14]) as hook_a:  #output hooks is calculated then gradient of each is calculated out
		with hook_output(model[14], grad = True) as hook_g:
			preds = model(b)
			preds[0,cls].backward()

	acts = hook_a.stored[0].cpu()
	grad = hook_g.stored[0][0].cpu()

	grad_chan = grad.mean(1).mean(1) #starndard technique i.e. avg gradient and add all to one
	mult = ((acts * grad_chan[...,None, None])).sum(0)

	_, ax = plt.subplots()
	img.show(ax)
	ax.imshow(mult, alpha = 0.4, extent=(0,256,256,0), interpolation='bicubic', cmap='jet')

@app.route('/',methods=['GET','POST'])
def home():
	if request.method == "POST":
		image_file = request.files["image"]
		if image_file:
			image_location = os.path.join(
				upload_folder,
				image_file.filename
				)
			image_file.save(image_location)

			image = cv2.imread(image_location)
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

			pil_im = PImage.fromarray(image)
			x = pil2tensor(pil_im, np.float32)
			# image_file = image_file.img_to_array(image_file)
			# t = torch.tensor(np.ascontiguousarray(image_file, dtype=np.float32)
			# img = Image(t)
			pred = make_prediction(x)
			# visualize(image_file)
			return render_template('startpage.html',prediction=pred)
	return render_template('startpage.html',prediction='Invalid')

if __name__ == "__main__":
    app.run(debug=True)