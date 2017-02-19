import cv2
import numpy as np
import os
import sys

def getList(base):
	for dirname, dirnames, filenames in os.walk(base):
		if dirname == base:
			return dirnames, filenames

def sub(a,b):
	res = []
	for i,j in zip(a,b):
		res.append(np.uint8(i-j))
	return res

def correct_gamma(img, correction):
    img = img/255.0
    img = cv2.pow(img, correction)
    return np.uint8(img*255)

def process_image(imag):  
    # gr = correct_gamma(imag, 3)
    gr = cv2.GaussianBlur(imag,(5,5),0)
    gr = cv2.equalizeHist(gr)
    r,c = np.where(gr < 200)
    gr[r,c]-=50
    r, gr = cv2.threshold(gr,127,255,cv2.THRESH_BINARY)
    return np.uint8(gr)

if __name__ == '__main__':
	dirs, x = getList('sample_drive')
	# print(dirs)
	dirs = ['cam_3']
	for d in dirs:
		x, files = getList('sample_drive/'+d)
		# print(len(files))
		masks = []
		# flag = np.ones(img.size,dtype=img.dtype)
		# flag *= 255
		
		dir = 'sample_drive/'+d
		tot = len(files)
		print(dir)
		c_masks = []
		count = 0
		avg = []
		avg_grad = []
		for i,f in enumerate(files):
			img = cv2.imread(dir+'/'+f,0)
			h,w = img.shape
			ratio = w/float(h)
			img = cv2.resize(img, (int(ratio*720), 720))
			# h,w = img.shape
			# new_h = int(h/3)
			# new_w = int(w/3)
			if i == 0:
				avg = np.zeros(img.shape,dtype=np.float64)
				avg_grad = np.zeros(img.shape,dtype=np.float64)
			avg_img = np.mean(img)
			if avg_img > 15:
				count += 1
				avg += np.float64(img)
				grad = cv2.Laplacian(img, cv2.CV_64F)
				avg_grad += grad
			# for x,im in enumerate(imgs):
				# imgs[x] = process_image(im)
			# img = cv2.equalizeHist(img)
			# img -= 70
			# for k in range(img.shape[0]):
			# 	for l in range(img.shape[1]):
			# 		if img[k,l] < 0 :
			# 			img[k,l] = 0
			flag = False
			

			# if len(masks) == 0:
			# 	masks = [np.ones(x.shape,dtype=x.dtype) for x in imgs]
			# else:
				
			# 	c_masks = sub(imgs,prev)
			# 	# if i%2 != 0:
			# 	# 	invert = lambda k: 255*np.ones(k.shape,dtype=k.dtype) - k
			# 	# 	c_masks = [invert(x) for x in c_masks]
			# 	temp = c_masks[:]
			# 	for x,c in enumerate(c_masks):
			# 		c_mask = ''
			# 		if i%2 == 0:
			# 			r, c_mask = cv2.threshold(c_masks[x], 127,255, cv2.THRESH_BINARY_INV)
			# 			flag = True
			# 		else:
			# 			r, c_mask = cv2.threshold(c_masks[x], 127,255, cv2.THRESH_BINARY)
			# 			flag = False
			# 		c_masks[x] = c_mask
			# 	masks = [cv2.bitwise_and(l,m) for l,m in zip(c_masks,masks)]
			# 	while True:
			# 		key = cv2.waitKey(10)
			# 		cv2.imshow('disp',c_masks[8])
			# 		cv2.imshow('image',imgs[8])
			# 		if key == 27:
			# 			break
			# prev = imgs[:]

			# if mask.__class__ is list:
			# 	mask = np.zeros(img.shape,dtype=img.dtype)
			# 	# ret, mask = cv2.threshold(mask, 50,255,cv2.THRESH_BINARY)
			# else:
			# 	c_mask = prev - img
			# 	ret,c_mask = cv2.threshold(c_mask, 100,255, cv2.THRESH_BINARY)
			# 	# mask = cv2.bitwise_or(mask,c_mask)
			# 	# mask+=img
			# 	# t = mask*255
			# 	while True:
			# 		key = cv2.waitKey(10)
			# 		cv2.imshow('disp',img)
			# 		if key == 27:
			# 			break
				# if np.array_equal(mask, flag):
					# break
				# print(mask[:10,:10])

			pro = ((i+1.)/tot)*100
			sys.stdout.write("\rCompleted: %.2f%% %d %d" % (pro,i,count))
			# sys.stdout.flush()
			# prev = img
		avg /= count
		avg_grad /= count
		avg = np.uint8(avg)
		avg_grad = np.uint8(avg_grad)
		while True:
			key = cv2.waitKey(10)
			if key == 27:
				break
			cv2.imshow('Average Image', avg)
			cv2.imshow('Average Grdient', avg_grad)
		# cv2.imwrite(d+'_mask.jpg',mask)
		print('\n')