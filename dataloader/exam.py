import os


def recheck(image_path):
	if not os.path.exists(image_path):
		file_path, img_name = os.path.split(image_path)
		info_list = img_name.split('.')
		img_num = int(info_list[0])
		image_path = os.path.join(file_path, str(img_num))
		image_path = image_path+'.'+info_list[1]
	return image_path

def rechecktest(image_path):
	if not os.path.exists(image_path):
		file_path, img_name = os.path.split(image_path)
		file_path, _ = os.path.split(file_path)
		info_list = img_name.split('.')
		info_list[0] = info_list[0][:-7]
		image_path = info_list[0]+'.'+info_list[1]
		image_path = os.path.join(file_path, image_path)
		
	return image_path