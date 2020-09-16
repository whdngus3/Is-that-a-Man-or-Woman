#1.이미지 생성(남자와 여자의데이터 100개씩을 생성하였지만 표본이 모자르기 때문에 기존데이터에 약간의 변화를준 새로운 데이터 생성)
#from keras.preprocessing.image import ImageDataGenerator
#datagen = ImageDataGenerator(
#                            rescale=1./255,
#                            zoom_range = [0.8,2.0],
#                            shear_range=0.5,
#                            rotation_range=15,
#                            horizontal_flip=True,
#                            vertical_flip=True,
#                            height_shift_range=0.1,
#                            width_shift_range=0.1,
#                            fill_mode = 'nearest')
#
#
#for i in range(images.shape[0]):
#    i=0
#    for batch in datagen.flow(images,batch_size=1,save_to_dir='D:\\man&women\\imgen_woman',save_prefix='1',save_format='jpg'):
#        i += 1
#        if i >10:
#            break
#


#2. 경로설정(자신의파일이있는곳을 설정해줘야함)

import glob
import numpy as np
import os.path as path
from scipy import misc

IMAGE_PATH = 'D:\man&women\수정'
file_paths = glob.glob(path.join(IMAGE_PATH, '*.jpg'))


file_paths[0] #잘 됐나 확인

#3. 이미지파일 불러오기
import imageio
images = [imageio.imread(path) for path in file_paths]

images = np.asarray(images)
images.shape


image_size = np.asarray([images.shape[1], images.shape[2], images.shape[3]])

images = images/255
images.shape

#4. 각 파일별 라벨만들어주기
n_images = images.shape[0]
labels = np.zeros(n_images)
for i in range(n_images):
    filename = path.basename(file_paths[i])[0]
    labels[i] = int(filename[0])


#5.트레인셋과 테스트셋설정
TRAIN_TEST_SPLIT=0.9

split_index = int(TRAIN_TEST_SPLIT*n_images)
shuffled_indices = np.random.permutation(n_images)
train_indices = shuffled_indices[0:split_index]
test_indices = shuffled_indices[split_index:]

x_train=images[train_indices,:,:]
y_train=labels[train_indices]
x_test=images[test_indices,:,:]
y_test=labels[test_indices]


#6.데이터가 잘 분류과 되었는지 확인(데이터 시각화)
def visualize_data(woman_images,man_images):
    figure=plt.figure()
    count=0
    
    for i in range(woman_images.shape[0]):
        count+=1
        figure.add_subplot(2,woman_images.shape[0],count)
        plt.imshow(woman_images[i,:,:])
        plt.axis('off')
        plt.title("1"),
        
        figure.add_subplot(1,woman_images.shape[0],count)
        plt.imshow(man_images[i,:,:])
        plt.axis('off')
        plt.title("0")
    
    plt.show()
    
    
N_TO_VISUALIZE=10

woman_example_indices=(y_train == 1)
woman_examples = x_train[woman_example_indices,:,:]
woman_examples = woman_examples[0:N_TO_VISUALIZE,:,:]

man_example_indices=(y_train == 0)
man_examples = x_train[man_example_indices,:,:]
man_examples = man_examples[0:N_TO_VISUALIZE,:,:]

visualize_data(woman_examples, man_examples)



#7.신경망 구축(4층)

import keras

from keras.models import Sequential
from keras.layers import Activation,Dropout,Flatten,Dense,Conv2D,MaxPooling2D
from keras.callbacks import EarlyStopping


def cnn(size,n_layers):
    MIN_NEURONS = 20
    MAX_NEURONS = 120
    KERNEL = (3,3)
    
    steps = np.floor(MAX_NEURONS/(n_layers +1))
    neurons = np.arange(MIN_NEURONS, MAX_NEURONS, steps)
    neurons = neurons.astype(np.int32)
    
    model = Sequential()
    
    for i in range(0,n_layers):
        if i == 0 :
            shape = (size[0],size[1],size[2])
            model.add(Conv2D(neurons[i],KERNEL,input_shape = shape))
        else:
            model.add(Conv2D(neurons[i],KERNEL))
            model.add(Activation('relu'))
            
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Flatten())
    model.add(Dense(MAX_NEURONS))
    model.add(Activation('relu'))
    
    # 출력층추가
    model.add(Dense(1,activation='sigmoid'))
    
    
    model.summary()
    
    return model

model = cnn(size=image_size,n_layers=N_LAYERS)

EPOCHS=15
BATCH_SIZE=20

PATIENCE = 100
early_stopping = EarlyStopping(monitor='loss',min_delta=0,patience=PATIENCE,verbose=0,mode='auto')

callbacks=[early_stopping]

model.compile(loss = 'binary_crossentropy',optimizer='adam',metrics = ['accuracy'])

#8. 모델 적합

model.fit(x_train, y_train,epochs=EPOCHS,batch_size=BATCH_SIZE,callbacks=callbacks,verbose=0)
evel = model.evaluate(x_test,y_test,batch_size=1)
print(evel)

#9.test데이터 결과(정확도)
p_xt = model.predict(x_test)
p_xt = np.round(p_xt)

import matplotlib.pyplot as plt
def visualize_incorrect_labels(x_data,y_real,y_predicted):
  count = 0
  figure = plt.figure()
  incorrect_label_indices = (y_real != y_predicted)
  y_real = y_real[incorrect_label_indices]
  y_predicted = y_predicted[incorrect_label_indices]
  x_data = x_data[incorrect_label_indices,:,:,:]

  maximum_square = np.ceil(np.sqrt(x_data.shape[0]))

  for i in range(x_data.shape[0]):
    count+=1
    figure.add_subplot(maximum_square, maximum_square,count)
    plt.imshow(x_data[i,:,:,:])
    plt.axis('off')
    plt.title("Predicted:"+str(int(y_predicted[i]))+',real:'+str(int(y_real[i])),fontsize=10)

  plt.show()
  print(x_data.shape[0])
visualize_incorrect_labels(x_test,y_test,np.asarray(p_xt).ravel())
res = np.asarray(p_xt).ravel()


###사진속인물의 성별맞추기

#onepick2
path = 'D:\man&women\test'
file_paths = glob.glob(path.join(IMAGE_PATH, '*.jpg'))

file_paths[-1]
x_pp = imageio.imread(file_paths[-1])
x_pp = np.asarray([x_pp])


plt.imshow(x_pp)
plt.show()

x_pp = x_pp/255
x_pp.shape

p_xt = model.predict(x_pp)
p_xt = np.round(p_xt)
res = np.asarray(p_xt).ravel()
if res[0]==1:
  print('woman')
else:
  print('man')
