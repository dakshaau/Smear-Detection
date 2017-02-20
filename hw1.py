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

# def ransac(n, d, k, t):

# 	pass

if __name__ == '__main__':
	dirs, x = getList('sample_drive')
	# print(dirs)
	# dirs = ['cam_3']
	for d in dirs:
		cv2.destroyAllWindows()
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
		# avg_grad = []
		# avg_blur = []
		if not os.path.exists(d+'_avg_img.jpg'): #or not os.path.exists(d+'_avg_blur.jpg'):
			for i,f in enumerate(files):
				img = cv2.imread(dir+'/'+f)
				h,w = img.shape[:2]
				ratio = w/float(h)
				img = cv2.resize(img, (int(ratio*720), 720))
				# h,w = img.shape
				# new_h = int(h/3)
				# new_w = int(w/3)
				if i == 0:
					avg = np.zeros(img.shape,dtype=np.float64)
					# avg_grad = np.zeros((img.shape[:2]),dtype=np.float64)
					# avg_blur = np.zeros((img.shape),dtype=np.float64)
				avg_img = np.mean(img)
				if avg_img > 15:
					count += 1
					# img_blur = cv2.blur(img,(3,3))
					# # while True:
					# # 	 k = cv2.waitKey(10)
					# # 	 if k == 27:
					# # 	 	break
					# # 	 cv2.imshow('blur',np.uint8(img_blur))
					# avg_blur += img_blur
					avg += np.float64(img)
					# img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
					# grad = cv2.Laplacian(img, cv2.CV_64F)
					# avg_grad += np.float64(grad)
				# for x,im in enumerate(imgs):
					# imgs[x] = process_image(im)
				# img = cv2.equalizeHist(img)
				# img -= 70
				# for k in range(img.shape[0]):
				# 	for l in range(img.shape[1]):
				# 		if img[k,l] < 0 :
				# 			img[k,l] = 0
					# flag = False
				

				pro = ((i+1.)/tot)*100
				sys.stdout.write("\rCompleted: %.2f%%" % (pro))
				# sys.stdout.flush()
				# prev = img
			avg /= count
			# avg_grad /= count
			# avg_blur /= count
			avg = np.uint8(avg)
			# avg_grad = np.uint8(avg_grad)
			# avg_blur = np.uint8(avg_blur)
			# while True:
			# 	key = cv2.waitKey(10)
			# 	if key == 27:
			# 		break
			# 	cv2.imshow('Average Image', avg)
			# 	cv2.imshow('Average Grdient', avg_grad)
			cv2.imwrite(d+'_avg_img.jpg',avg)
			# cv2.imwrite(d+'_avg_grad.jpg',avg_grad)
			# cv2.imwrite(d+'_avg_blur.jpg',avg_blur)
			print('\n')
		
		img = cv2.imread(d+'_avg_img.jpg',0)
		# grad = cv2.imread(d+'_avg_grad.jpg',0)
		# blur = cv2.imread(d+'_avg_blur.jpg',0)
		imgi = np.int32(img)

		# bluri = np.int32(blur)
		r,c = np.where(imgi < 100)
		imgi[r,c] = 0
		r,c = np.where(imgi >= 100)
		imgi[r,c] = 255
		imgi = np.uint8(imgi)

		h = imgi.shape[0]
		new_h = int(h/3)
			
		# Since we know that the center part of the image consists of patterns from the road
		# for our currrent dataset. So we can simply exclude the center part of the image from
		# smear mask

		imgi[new_h:2*new_h,:] = 255

		# imgi = cv2.threshold(imgi, 127, 255, cv2.THRESH_BINARY_INV)
		mask = 255*np.ones(imgi.shape,dtype=imgi.dtype) - imgi
		kernel = np.ones((5,5), np.uint8)
		mask = cv2.dilate(mask, kernel, iterations = 1)
		r,c = np.where(mask == 0)

		img1 = img.copy()

		img1[r,c] = 0

		# mask = bluri - imgi
		# mask = np.uint8(mask)
		cv2.imwrite(d+'_smear_mask.jpg',mask)

		while True:
			key = cv2.waitKey(10)
			if key == 27:
				cv2.destroyAllWindows()
				break
			cv2.imshow('Mask',mask)
			cv2.imshow('Smear',img1)
			cv2.imshow('Average Image', img)
