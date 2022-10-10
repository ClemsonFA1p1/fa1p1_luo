"""
Script to generate video from a set of images in an episode

data_type: folder containing the data
episode_number: specific episode number containing the images
"""

from PIL import Image, ImageFont, ImageDraw
import csv
import os
import sys
import json
import numpy as np
import skvideo
skvideo.setFFmpegPath('/home/bsande6/anaconda3/envs/coiltraine/bin/')
#skvideo.setFFmpegPath('/home/bsande6/anaconda3/envs/coiltraine/lib/python3.5/site-packages/skvideo/io/')
import skvideo.io

DATA_DIR = '_benchmarks_results/left_style_iteration_100000_carla_LtoG_iteration_500000.ptnocrash_resnet34imnet10S1_660000_drive_control_output_NocrashTraining_Town01/_images'


if __name__ == '__main__':
	#data_type = sys.argv[1]
	episode_number = sys.argv[1]
	episode_path = os.path.join(DATA_DIR,'episode_%s'%(episode_number))
	print(os.path.exists(episode_path))
	rgb_path = os.path.join(episode_path, "rgb")
	print(os.path.exists(rgb_path))
	translated_path = os.path.join(episode_path, "translation")
	print(os.path.exists(translated_path))
	#episode_data = sorted(os.listdir(episode_path))
	translated_data = sorted(os.listdir(translated_path))
	rgb_data = sorted(os.listdir(episode_path))

	measurements_data = []
	expert_steer = []
	expert_throttle = []
	expert_brake = []
	agent_steer = []
	agent_throttle = []
	agent_brake = []
	directions = []
	speed_module = []
	rgb_image_path = []
	translated_image_path=[]
	

	for file_name in rgb_data:
		if 'rgb' in file_name:
			rgb_cam_data = sorted(os.listdir(os.path.join(episode_path, file_name)))
			for image in rgb_cam_data:
				image_path = os.path.join(episode_path, file_name, image)
				print(image_path)
				rgb_image_path.append(image_path)
		else:
			print(file_name)
			translated_data = sorted(os.listdir(os.path.join(episode_path, file_name)))
			for image in translated_data:
				image_path = os.path.join(episode_path, file_name, image)
				print(image_path)
				translated_image_path.append(image_path)


	# print (len(measurements_data), len(central_image_path), len(expert_steer), len(expert_throttle), len(expert_brake),
	# 	   len(agent_steer), len(agent_throttle), len(agent_brake), len(directions), len(speed_module))
	if not os.path.isdir(os.path.join(DATA_DIR, 'videos')):
		os.mkdir(os.path.join(DATA_DIR, 'videos'))
	# img = Image.open(left_image_path[1])
	# img = np.asarray(img)
	# print(img.shape)
	# skvideo.io.vwrite(os.path.join(DATA_DIR, 'videos', '%s_episode_%s.mp4'%(data_type, episode_number)), img)
	# vid = skvideo.io.vread(left_image_path[1])
	# T, M, N, C = vid.shape

	# print("Number of frames: %d" % (T,))
	# print("Number of rows: %d" % (M,))
	# print("Number of cols: %d" % (N,))
	
	writer = skvideo.io.FFmpegWriter(os.path.join(DATA_DIR, 'videos', '%s_episode_%s.mp4'%("_images", episode_number)), inputdict={'-r': '10', '-s':'800x600'},
            outputdict={'-r': '10',  '-pix_fmt': 'yuv420p'})
	for i in range(1, len(rgb_image_path)):
		img = Image.open(rgb_image_path[i])
		translated_img = Image.open(translated_image_path[i])
		#center_img = np.asarray(center_img)
		#center_img = np.resize(center_img, (100, 120))
		translated_img = translated_img.resize((200, 120), resample=0)
		img.paste(translated_img, (30,40, 230, 160))

		helvetica = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSerif.ttf", size=20)
		d = ImageDraw.Draw(img)
		text_color = (255, 255, 255)

		location = (40, 10)
		d.text(location, "Synthesized Center Camera", font=helvetica, fill=text_color)

		# location = (40, 35)
		# d.text(location, "agent_throttle = %0.4f"%agent_throttle[i], font=helvetica, fill=text_color)

		# location = (40, 60)
		# d.text(location, "agent_brake = %0.4f"%agent_brake[i], font=helvetica, fill=text_color)

		location = (300, 10)
		d.text(location, "Aerial Camera",  font=helvetica, fill=text_color)

		# location = (300, 35)
		# d.text(location, "expert_throttle = %0.4f"%expert_throttle[i], font=helvetica, fill=text_color)

		location = (600, 60)
		#image.paste(xy, color, bitmap)
		#d.rectangle(location, fill=center_img, outline=None, width=1)
		# d.text(location, "expert_brake = %0.4f"%expert_bake[i], font=helvetica, fill=text_color)

		# location = (40, 85)
		# d.text(location, "vehicle_speed = %0.4f"%speed_module[i], font=helvetica, fill=text_color)
		img = np.asarray(img)
		# pilImage = Image.fromarray(numpydata)
		writer.writeFrame(img)

	writer.close()
