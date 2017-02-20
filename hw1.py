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
		if not os.path.exists(d+'_avg_img.jpg'):
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
					avg_grad = np.zeros((img.shape[:2]),dtype=np.float64)
				avg_img = np.mean(img)
				if avg_img > 15:
					count += 1
					avg += np.float64(img)
					img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
					grad = cv2.Laplacian(img, cv2.CV_64F)
					avg_grad += np.float64(grad)
				# for x,im in enumerate(imgs):
					# imgs[x] = process_image(im)
				# img = cv2.equalizeHist(img)
				# img -= 70
				# for k in range(img.shape[0]):
				# 	for l in range(img.shape[1]):
				# 		if img[k,l] < 0 :
				# 			img[k,l] = 0
					flag = False
				

				pro = ((i+1.)/tot)*100
				sys.stdout.write("\rCompleted: %.2f%% %d %d" % (pro,i,count))
				# sys.stdout.flush()
				# prev = img
			avg /= count
			avg_grad /= count
			avg = np.uint8(avg)
			avg_grad = np.uint8(avg_grad)
			# while True:
			# 	key = cv2.waitKey(10)
			# 	if key == 27:
			# 		break
			# 	cv2.imshow('Average Image', avg)
			# 	cv2.imshow('Average Grdient', avg_grad)
			cv2.imwrite(d+'_avg_img.jpg',avg)
			cv2.imwrite(d+'_avg_grad.jpg',avg_grad)
			print('\n')
		else:
			img = cv2.imread(d+'_avg_img - Copy.jpg',0)
			grad = cv2.imread(d+'_avg_grad.jpg',0)

			imgi = np.int32(img)
			r,c = np.where(imgi < 90)
			imgi[r,c] -= 40
			r,c = np.where(imgi >= 90)
			imgi[r,c] += 40
			imgi = np.uint8(imgi)

			while True:
				key = cv2.waitKey(10)
				if key == 27:
					break
				cv2.imshow('Dilated Gradient',imgi)
				# cv2.imshow('Gradient',im)
