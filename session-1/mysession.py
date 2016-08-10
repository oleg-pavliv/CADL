# coding: utf-8
get_ipython().magic('pylab')
get_ipython().magic('ls')
# example of magic functions
get_ipython().magic('paste')
def say_hello():
      print ("hello")
    
get_ipython().magic('ls')
plt.style.use('ggplot')
from libs import utils

utils.remove_duplicates('../daisy/')

#example of reload
import importlib
importlib.reload(utils)

import os
files = [f for f in os.listdir('../daisy') if '.jpg' in f]
len(files)
images = [plt.imread('../daisy/' + f) for f in files]
images_sq = [utils.imcrop_tosquare(i) for i in images]

min_img_sq = min(images_sq, key = lambda im: im.shape[1])
min_img_sq.shape
plt.imshow(min_img_sq[:,:,0], cmap='gray')
plt.imshow(min_img_sq[:,:,1], cmap='gray')
plt.imshow(min_img_sq[:,:,2], cmap='gray')

from scipy.misc import imresize

images_min = [imresize(im, (177, 177)) for im in images_sq]

data_min = np.array(images_min)
data_min.shape

mean_daisy = np.mean(data_min, axis=0).astype(np.uint8)
plt.imshow(mean_daisy)
std_daisy = np.std(data_min, axis=0).astype(np.uint8)
plt.imshow(std_daisy, cmap='gray')

plt.imshow(np.mean(data_min, axis=2).astype(np.uint8))
plt.imshow(np.mean(std_daisy, axis=2).astype(np.uint8))
plt.imshow(np.mean(std_daisy, axis=2).astype(np.uint8), cmap='gray')
plt.imshow(np.mean(std_daisy, axis=2).astype(np.uint8), cmap='gray')
plt.imshow(np.mean(std_daisy, axis=0).astype(np.uint8), cmap='gray')
plt.imshow(np.mean(std_daisy, axis=1).astype(np.uint8), cmap='gray')
plt.imshow(np.mean(std_daisy, axis=2).astype(np.uint8), cmap='gray')

im33_0 = array([[[0,255,0],[0,255,0],[0,255,0]],[[0,255,0],[0,255,0],[0,255,0]],[[0,255,0],[0,255,0],[0,255,0]]], dtype=uint8)
im33_1 = array([[[255,0,0],[0,255,0],[0,255,0]],[[0,255,0],[0,255,0],[0,255,0]],[[0,255,0],[0,255,0],[0,255,0]]], dtype=uint8)
im33_2 = array([[[255,0,0],[255,0,0],[0,255,0]],[[0,255,0],[0,255,0],[0,255,0]],[[0,255,0],[0,255,0],[0,255,0]]], dtype=uint8)
im33_3 = array([[[255,0,0],[0,255,0],[0,255,0]],[[255,0,0],[0,255,0],[0,255,0]],[[0,255,0],[0,255,0],[0,255,0]]], dtype=uint8)
im33_4 = array([[[255,0,0],[255,0,0],[0,255,0]],[[255,0,0],[0,255,0],[0,255,0]],[[0,255,0],[0,255,0],[0,255,0]]], dtype=uint8)
im33_5 = array([[[255,0,0],[0,255,0],[0,255,0]],[[0,255,0],[255,0,0],[0,255,0]],[[0,255,0],[0,255,0],[0,255,0]]], dtype=uint8)
im33=[im33_1, im33_2, im33_3, im33_4, im33_5]
data33 = np.array(im33)
data33.shape
np.mean(data33, axis=0).astype(np.uint8)
plt.imshow(np.mean(data33, axis=0).astype(np.uint8))
np.std(data33, axis=0).astype(np.uint8)
plt.imshow(np.std(data33, axis=0).astype(np.uint8))
plt.imshow(np.std(data33, axis=0).astype(np.uint8), cmap='gray')
plt.imshow(np.std(data33, axis=0).astype(np.uint8), cmap='hot')
plt.imshow(np.std(data33, axis=0).astype(np.uint8))
mean33=np.mean(data33, axis=0).astype(np.uint8)
mean33
std33=np.std(data33, axis=0).astype(np.uint8)
std33
np.mean(std33,axis=2)
plt.imshow(np.mean(std33,axis=2).astype(uint8), cmap='gray')
plt.imshow(np.mean(std33,axis=0).astype(uint8), cmap='gray')
plt.imshow(np.mean(std33,axis=1).astype(uint8), cmap='gray')
plt.imshow(np.mean(std33,axis=2).astype(uint8), cmap='gray')

np.mean(std33, axis=2)
plt.imshow(std33)
plt.imshow(std33[0,:,:])
plt.imshow(std33[1,:,:])
plt.imshow(std33[1,:,:])
plt.imshow(std33[:,:,0])
plt.imshow(std33[:,:,1])
plt.imshow(std33[:,:,2])

data_daisy = data_min
flattened_daisy = data_daisy.ravel()
plt.hist(flattened_daisy, 255)

bins =20
plt.clean('all')
fig,axs = plt.subplots(1, 3, figsize=(12, 6), sharey=True, sharex=True)
axs[0].hist((data_daisy[0]).ravel(), bins)
axs[1].hist((mean_daisy).ravel(), bins)
axs[2].hist((data_daisy[0] - mean_daisy).ravel(), bins)

plt.figure()
plt.imshow((data_daisy[5]-mean_daisy)/std_daisy)
plt.imshow((data_daisy[6]-mean_daisy)/std_daisy)
plt.imshow((data_daisy[7]-mean_daisy)/std_daisy)

import tensorflow as tf
sess = tf.InteractiveSession()
z = (tf.exp(tf.neg(tf.pow(x - mean, 2.0)/(2.0 * tf.pow(sigma, 2.0))))/(sigma*tf.sqrt(2.0*3.1415)))

plt.figure()
timg1 = array([[1.0, 0, 0, 0], [2.0, 0, 0, 0],[3.0, 0, 0, 0],[44.0, 0, 0, 0]], dtype=float32)
plt.imshow(timg1, cmap='gray')
timg1 = array([[1.0, 0, 0, 0], [2.0, 0, 0, 0],[3.0, 0, 0, 0],[4.0, 0, 0, 0]], dtype=float32)
plt.imshow(timg1, cmap='gray')
plt.imshow(timg1, cmap='gray')
timg1 = array([[1.0, 0, 0, 0], [2.0, 0, 0, 0],[3.0, 0, 0, 0],[-4.0, 0, 0, 0]], dtype=float32)
plt.imshow(timg1, cmap='gray')
timg1 = array([[1.0, 0, 0, 0], [2.0, 0, 0, 0],[-33.0, 0, 0, 0],[-4.0, 0, 0, 0]], dtype=float32)
plt.imshow(timg1, cmap='gray')

img=tf.placeholder(tf.float32, shape=[None, None], name='img')
img3d=tf.expand_dims(img,2)
img4d=tf.expand_dims(img3d,0)
img4d.get_shape()
mean = tf.placeholder(tf.float32, name='mean')
sigma = tf.placeholder(tf.float32, name='sigma')
ksize = tf.placeholder(tf.int32, name='ksize')
x = tf.linspace(-3.0, 3.0, ksize)
z = (tf.exp(tf.neg(tf.pow(x - mean, 2.0)/(2.0*sigma*sigma))))/(sigma*tf.sqrt(2.0*3.1415))
z_2d = tf.matmul(tf.reshape(z, tf.pack([ksize, 1])), tf.reshape(z, tf.pack([1,ksize])))
ys = tf.sin(x)
ys = tf.reshape(ys, tf.pack([ksize, 1]))
ones = tf.ones(tf.pack([1, ksize]))
wave = tf.matmul(ys, ones)
gabor = tf.mul(wave, z_2d)
gabor_4d  = tf.reshape(gabor, tf.pack([ksize, ksize, 1, 1]))
convolved = tf.nn.conv2d(img4d, gabor_4d, strides = [1,1,1,1], padding='SAME', name='convolved')
convolved_img = convolved[0, : , :, 0]
res = convolved_img.eval(feed_dict={img:data.camera(), mean:0.0, sigma:1.0, ksize:10})
plt.imshow(res, cmap='gray')
res = convolved_img.eval(feed_dict={img:data.camera(), mean:0.0, sigma:0.5, ksize:10})
plt.imshow(res, cmap='gray')
res = convolved_img.eval(feed_dict={img:data.camera(), mean:0.0, sigma:0.5, ksize:5})
plt.imshow(res, cmap='gray')

