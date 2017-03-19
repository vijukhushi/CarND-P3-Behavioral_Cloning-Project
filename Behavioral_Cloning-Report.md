{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {
    "deletable": true,
    "editable": true
   },
   "source": [
    "### **CarND-Behavioral-Cloning-P3**\n",
    "**Behavioral-Cloning**\n",
    "\n",
    "This is the third project by me in the field of Deep Mchine learning as part of Udacity Nanodegree for Autonomous driving.\n",
    "\n",
    "While the pervious project was on 'Logistic Classification' of the Deep Neural network model, this one is on the 'Regression' Supervised learning of the Deep Neural network.\n",
    "\n",
    "The objective of this project(Behavioral Cloning) is to train a model to drive a car autonomously on a simulated track. The ability of the model to drive the car autonomously is learned from cloning the behaviour of a human driver. Training data is collected from examples of a human driving in the simulator provided by the Udacity, then fed into a deep learning network which learns the response (steering angle) for every encountered frame in the simulation. In other words, the model is trained to predict an appropriate steering angle for every frame while driving(hence Regression). The model is then validated on a new track to check for generalization of the learned features for performing steering angle prediction.\n",
    "\n",
    "This project is influenced by nvidia paper, Udacity Lab sessions and vivek's blog which I reffered while working on my solution. The Keras Deep Learning library was used with Tensorflow backend, to perform deep learning operations."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "deletable": true,
    "editable": true
   },
   "source": [
    "**Behavioral Cloning Project Steps**\n",
    "\n",
    "In Summary, the goals / steps of this project are the following:\n",
    "* Use the simulator to collect data of good driving behavior\n",
    "* Build, a convolution neural network in Keras that predicts steering angles from images\n",
    "* Train and validate the model with a training and validation set\n",
    "* Test that the model successfully drives around track one without leaving the road\n",
    "* Summarize the results with a written report"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "deletable": true,
    "editable": true
   },
   "source": [
    "### Data Recording"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "I can either use the data of steering angles and images provided by Udacity, or collect data ourselves by driving a car around a track in Udacity's simulator. \n",
    "\n",
    "Upon analysis of the sample data provided, the sample data has \"Orginal Data Size: 8036\". By ploting the graph on the streeing angle and # of images per streeing angle, we can see that there is more data with steering angle on the left."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "![Data Histogram](images/DataHistogram.png)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "deletable": true,
    "editable": true
   },
   "source": [
    "\n",
    "\n",
    "This is true from the way the track is designed that the Track #1 is more heavy on the left turn than right.\n",
    "So the data is not balanced equally. Also the # of sample data is not sufficient to train the model without under fitting or over fitting the model with just images from center camera and streeing angle.\n",
    "\n",
    "Hence this has been solved by following the below methods i.e Collecting more data, preprocessing and data augumentation.\n",
    "\n",
    "Data Collection:\n",
    "In order to collect more data and more over balanced data, I drove the car in 'Training Mode' where we collect the data.\n",
    "#1. I did around 2 laps of driving to supply enough images and corresponding steering angles to work with.\n",
    "#2. 1 round of lap driving in the opposite direction (clockwise) to collect more data in order to negate the left baised track\n",
    "#3. Drove the car to the edge of the track without recording and corrected back to the center of the road with recording enabled. This will train the model to correct the car or bring it back to the center of the road if it drifts off the road.\n",
    "\n",
    "Once I had my own recorded data, I used the sample data for validation. This way I could use all the data I had collected to train the model and validate the model aganist the sample data provided."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "deletable": true,
    "editable": true
   },
   "source": [
    "Training Image:\n",
    "![Training Image](images/Training.png)\n",
    "\n",
    "Reverse Lap Training Image:\n",
    "![Reverse Lap](images/ReverseDir.png)\n",
    "\n",
    "Edge Recovery Image:\n",
    "![Edge Correction](images/edgeCorrection.png)\n",
    "\n",
    "Dirt Pit Recovery Image:\n",
    "![Dirt Correction](images/DirtPitCorrection.png)\n",
    "\n",
    "Corner/Curve Recovery Image:\n",
    "![Corner Correction](images/cornercorrection.png)\n",
    "\n",
    "Bridge Edge Recovery Image:\n",
    "![Bridge Correction](images/BridgeEdgeCorrection.png)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "deletable": true,
    "editable": true
   },
   "source": [
    "Once I had the data captured, I loaded the data of around ~15K images from the csv file. The images are in three colums, snapped from center, right and left cameras. They have a corresponding steering angle -- the steer the car had to make in that frame to stay on the track. I loaded both into images[] and measurements[] arrays. \n",
    "\n",
    "Once I got the steering angles, I increased the value of the steering angle by 20% (multiplied by 1.2) to make the car more responsive. The main reason to do so was while training I went over the edges a few times, and corrected. I wanted the car to avoid those mistakes, so used higher gain on steering angle."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true,
    "deletable": true,
    "editable": true
   },
   "outputs": [],
   "source": [
    "# Load CSV File for Validation Data       \n",
    "#Get Validation Data (This is Udacity provided same data)\n",
    "Val_lines = []\n",
    "with open('Val_DATA/driving_log.csv') as csvfile2 :\n",
    "    reader = csv.reader(csvfile2)\n",
    "    for line in reader :\n",
    "        Val_lines.append(line)\n",
    "        \n",
    "Val_images = [] # Validation featuers\n",
    "Val_measurments = [] # Validation labels\n",
    "\n",
    "for line in Val_lines :\n",
    "    #print(line[0])\n",
    "    Val_source_path = line[0] # to get the center image file name/paths\n",
    "    Val_file_name = Val_source_path.split('/')[-1]\n",
    "    #print(file_name)\n",
    "    Val_current_path = 'Val_DATA/IMG/'+Val_file_name\n",
    "    Val_image = cv2.imread(Val_current_path)\n",
    "    Val_images.append(Val_image)\n",
    "    Val_measurment = float(line[3]) # to get the steering angle from the csv file\n",
    "    Val_measurments.append(Val_measurment)\n",
    "    \n",
    "X_Val = np.array(Val_images)\n",
    "y_Val = np.array(Val_measurments)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true,
    "deletable": true,
    "editable": true
   },
   "outputs": [],
   "source": [
    "Orginal Training Data Size: 14642\n",
    "Orginal Validation Data Size: 8036"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "deletable": true,
    "editable": true
   },
   "source": [
    "### Data processing"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "deletable": true,
    "editable": true
   },
   "source": [
    "Data processing is done to allow the model to be able to easily work with raw data for training. In this project, I used a generator (keras fit_generator) to build the data processing as it allows for real-time processing of the data without consuming more memory or CPU. The advantage here is that, in the case that if we are working with a very large amount of data, the whole dataset is not loaded into memory at one shot, instead processed batch wise. We can therefore work with a manageable batch of data at a time. Hence the generator is run in parallel to the model, for efficiency. The generator is run in CPU per image, processed and sent to Keras model which is run on Tensorflow in GPU\n",
    "\n",
    "The following processing steps were carried on the data:\n",
    "\n",
    "*Randomly choose from center, left and right camera images:\n",
    "\n",
    "The simulator provides three camera views namely; center, left and right views. Since I am going to use the generator which requires to use only one camera view per generator, I choose randomly from the three views. While using the left and right images, I did add and subtract 0.25 to the steering angles respectively to make up for the camera offsets (Tried with few other values, but felt 0.25 provided best results in terms of stability of the car).\n",
    "\n",
    "*Randomly flip image:\n",
    "\n",
    "In other to balance left and right images, I randomly flip images and change sign on the steering angles. The following figure shows the view from the left, right and center cameras after angles corrected. \n",
    "\n",
    "*Brightness Augmentation:\n",
    "\n",
    "We simulate different brightness occasions by converting image to HSV channel and randomly scaling the V channel.\n",
    "\n",
    "*Image Cropping: \n",
    "\n",
    "The top part of each image has all the trees and senery and bottom part has the car hood. These are not required for the model and may cause unnessary confusion while traning. Hence Cropped the top 60 pixels and bottom 20 pixels using the Keras Cropping function so as to feed the data that is required for the model to train and learn the behavior.\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true,
    "deletable": true,
    "editable": true
   },
   "outputs": [],
   "source": [
    "\n",
    "#simulate different brightness occasions by converting image to HSV channel and randomly scaling the V channel.\n",
    "def transform_brightness(image):\n",
    "    \"\"\"\n",
    "    apply random brightness on the image\n",
    "    \"\"\"\n",
    "    image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)\n",
    "    random_bright = .25+np.random.uniform()\n",
    "    \n",
    "    # scaling up or down the V channel of HSV\n",
    "    image[:,:,2] = image[:,:,2]*random_bright\n",
    "    return image\n",
    "\n",
    "## Data Augmuntation (Flip Images and steering angle)\n",
    "augmented_images, augmented_angles = [], []\n",
    "def data_augmuntation(images, angles):\n",
    "    for image, angle in zip(images, angles):\n",
    "        augmented_images.append(image)\n",
    "        augmented_angles.append(angle)\n",
    "        augmented_images.append(cv2.flip(image,1))\n",
    "        augmented_angles.append(angle*-1.0)\n",
    "    return augmented_images, augmented_angles\n",
    "\n",
    "def data_flip(image, angle):\n",
    "    augmented_images.append(image)\n",
    "    augmented_angles.append(angle)\n",
    "    # flip image (randomly)\n",
    "    if np.random.randint(2) == 0: \n",
    "        augmented_images.append(cv2.flip(image,1))\n",
    "        augmented_angles.append(angle*-1.0)\n",
    "    return augmented_images, augmented_angles\n",
    "\n",
    "# Load the data\n",
    "images = [] # featuers\n",
    "measurments = [] # labels\n",
    "\n",
    "for line in lines :\n",
    "    for i in range(3) :\n",
    "        #print(line[0])\n",
    "        source_path = line[i] # to get the center, right and left image file name/paths\n",
    "        file_name = source_path.split('/')[-1]\n",
    "        #print(file_name)\n",
    "        current_path = 'Val_DATA/IMG/'+file_name\n",
    "        image = cv2.imread(current_path)\n",
    "        #image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)\n",
    "        image = transform_brightness(image)\n",
    "        images.append(image) # Images from Center, Left and Right camera are appended into 'images'\n",
    "    correction = 0.20\n",
    "    measurment = float(line[3]) # to get the steering angle from the csv file\n",
    "    measurments.append(measurment) # Steering angle for Center Image\n",
    "    measurments.append(measurment + correction) # Steering angle for Left Image\n",
    "    measurments.append(measurment - correction) # Steering angle for Right Image\n",
    "    \n",
    "augmented_images, augmented_angles = data_augmuntation(images, measurments)\n",
    "X_train = np.array(augmented_images)\n",
    "y_train = np.array(augmented_angles)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "deletable": true,
    "editable": true
   },
   "source": [
    "### Model Training"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "deletable": true,
    "editable": true
   },
   "source": [
    "I Selected Nvidia model for final training, because it gave better result after experimenting with other kinds of model (e.g. LeNet, simple models from Lab sessions etc). The Nvidia network consists of 9 layers,which has 5 convolutional layers and 4 fully connected layers Plus a normalization layer and Cropping of image. Converse to the Nvidia model, input image was split to HSV planes before been passed to the network.\n",
    "\n",
    "Each of the layer is activitated by RELU activation.\n",
    "\n",
    "Below is the reference for the Nvidia architecture guidence.\n",
    "\n",
    "https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "deletable": true,
    "editable": true
   },
   "source": [
    "![Nvidia Arch](img/NvidiaArch.png)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "collapsed": true,
    "deletable": true,
    "editable": true
   },
   "source": [
    "Image was normalized in the first layer. According to the Nvidia paper, this enables normalization also to be accelerated via GPU processing.\n",
    "\n",
    "Convolution are used in the first three layers with 2x2 strides and a 5x5 kernel, and non-strided convolution with 3x3 kernel size in the last two convolutional layers.\n",
    "\n",
    "The convolutional layers are followed by three fully-connected layers which then outputs/predicts the steering angle.\n",
    "\n",
    "Overfitting was reduced by using aggressive Dropout (0.3) on all the Convolution layers and L2 Regularization (0.001) on the first layer. This turned out to be a good practice as the model could generalize to the second track, without using it for training.\n",
    "\n",
    "An Adam optimizer was used for optimization. This requires little or no tunning as the learning rate is adaptive. I run the training model for 3 Epochs as I could see that after it the loss is pretty much flat.\n",
    "\n",
    "Though I had close to 15K sample training data, each epoch consisted of twice the samples (samples_per_epoch=len(train_samples)*2). This made the training more tractable, and since I was using a generator, all of the training data was still used in training, however at different epochs.\n",
    "\n",
    "To conclude, self-recorded data was used for training, while the Udacity-provided samle data was used for validation during training."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true,
    "deletable": true,
    "editable": true
   },
   "outputs": [],
   "source": [
    "# Nvidia Model used for Behavioural Cloning Project\n",
    "def nvidia_model(time_len=1):\n",
    "    keep_prob = 0.3\n",
    "    reg_val = 0.01\n",
    "    model = Sequential()\n",
    "    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))\n",
    "    model.add(Cropping2D(cropping=((60,20),(0,0)))) # Cropping the top 60 and bottom 20 pixels of the images\n",
    "    model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu', W_regularizer=l2(reg_val)))\n",
    "    model.add(Dropout(keep_prob))\n",
    "    model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))\n",
    "    model.add(Dropout(keep_prob))\n",
    "    model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))\n",
    "    model.add(Dropout(keep_prob))\n",
    "    model.add(Convolution2D(64,3,3,activation='relu'))\n",
    "    model.add(Dropout(keep_prob))\n",
    "    model.add(Convolution2D(64,3,3,activation='relu'))\n",
    "    model.add(Flatten())\n",
    "    model.add(Dense(120))\n",
    "    model.add(Activation('relu'))\n",
    "    model.add(Dense(84))\n",
    "    model.add(Activation('relu'))\n",
    "    model.add(Dense(10))\n",
    "    model.add(Activation('relu'))\n",
    "    model.add(Dense(1))\n",
    "    model.compile(optimizer='adam', loss='mse')\n",
    "    return model"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "deletable": true,
    "editable": true
   },
   "source": [
    "### Training and validation"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "deletable": true,
    "editable": true
   },
   "source": [
    "The Above Nvidia was the final model I selected to train the model. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true,
    "deletable": true,
    "editable": true
   },
   "outputs": [],
   "source": [
    "# Select a Model\n",
    "model = nvidia_model()\n",
    "#model = Lenet_model()\n",
    "\n",
    "\n",
    "history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples)*2, validation_data=(X_Val, y_Val), nb_epoch=3)\n",
    "model.save('model.h5')\n",
    "\n",
    "### print the keys contained in the history object\n",
    "print(history_object.history.keys())\n",
    "\n",
    "### plot the training and validation loss for each epoch\n",
    "plt.plot(history_object.history['loss'])\n",
    "plt.plot(history_object.history['val_loss'])\n",
    "plt.title('model mean squared error loss')\n",
    "plt.ylabel('mean squared error loss')\n",
    "plt.xlabel('epoch')\n",
    "plt.legend(['training set', 'validation set'], loc='upper right')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "deletable": true,
    "editable": true
   },
   "source": [
    "I trained the model using the keras generator with batch size of 128 for 3 epochs. In each epoch, I generated len(train_samples)*2 images. The entire training took about 5 -7 minutes. However, it too more than few days of trial and error method to arrive at this architecture and training parameters. Snippet below presents the result of training."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "deletable": true,
    "editable": true
   },
   "source": [
    "Orginal Training Data Size: 6428\n",
    "Orginal Validation Data Size: 1608\n",
    "no_of_epochs=:Epoch 1/3 \n",
    "50.21875\n",
    "12821/12856 [============================>.] - ETA: 0s - loss: 0.0112   \n",
    "/home/carnd/anaconda3/envs/carnd-term1/lib/python3.5/site-packages/keras/engine/training.py:1569: UserWarning: Epoch comprised more than `samples_per_epoch` samples, which might affect learning results. Set `samples_per_epoch` correctly to avoid this warning.\n",
    "  warnings.warn('Epoch comprised more than '\n",
    "13017/12856 [==============================] - 69s - loss: 0.0112 - val_loss: 0.0180\n",
    "Epoch 2/3\n",
    "12910/12856 [==============================] - 64s - loss: 0.0053 - val_loss: 0.0193\n",
    "Epoch 3/3\n",
    "13052/12856 [==============================] - 55s - loss: 0.0047 - val_loss: 0.0211\n",
    "dict_keys(['val_loss', 'loss'])\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "deletable": true,
    "editable": true
   },
   "source": [
    "Once I had the best model that I thought will work, then I downloanded the saved mode (model.h5) to the local system to test it out on the simulator.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true,
    "deletable": true,
    "editable": true
   },
   "outputs": [],
   "source": [
    "\n",
    "scp carnd@54.1.1.1:/home/carnd/CarND-Behavioral-Cloning-P3/model.h5 ."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "deletable": true,
    "editable": true
   },
   "source": [
    "Run the below commad to test the model and at the same time save the model o/p to a file to record the simulation."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true,
    "deletable": true,
    "editable": true
   },
   "outputs": [],
   "source": [
    "(carnd-term1) $ python drive.py model_Final_Nvida_Bri.h5 Final_Nvida_Bri"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "deletable": true,
    "editable": true
   },
   "source": [
    "As expected, the car was able to drive and most part of the track #1, it was able to stay close to the center of the road. Once I had the car drive by itself on Track #1 for one full lap (Man was I happy..)exited out of the simulator and generated the video by using the below Udacity provided script - "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true,
    "deletable": true,
    "editable": true
   },
   "outputs": [],
   "source": [
    "(carnd-term1):CarND-Behavioral-Cloning-P3 vijaynaik$ python video.py Final_Nvida_Bri\n",
    "Creating video Final_Nvida_Bri.mp4, FPS=60\n",
    "[MoviePy] >>>> Building video Final_Nvida_Bri.mp4\n",
    "[MoviePy] Writing video Final_Nvida_Bri.mp4\n",
    "100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5539/5539 [00:27<00:00, 199.58it/s]\n",
    "[MoviePy] Done.\n",
    "[MoviePy] >>>> Video ready: Final_Nvida_Bri.mp4\n",
    "\n",
    "(carnd-term1) nss-admins-MacBook-Pro-6:CarND-Behavioral-Cloning-P3 vijaynaik$"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "deletable": true,
    "editable": true
   },
   "source": [
    "All the required files, this report, images and scripts have been submited as part of project submission package."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "deletable": true,
    "editable": true
   },
   "source": [
    "### Conclusion"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "deletable": true,
    "editable": true
   },
   "source": [
    "At first it looked simple and straight forward when I was doing all the lab session and thought would be able to complet it soon.\n",
    "\n",
    "As I started exploring the data and started implementing all the required modules, it is then when it started getting complicated. I had to spend close to a week in trying to build various different types of models, trying various different hyper parameters.\n",
    "\n",
    "At times it was fasturating as well to get the model work for just track #1 itself. I had to rebuild the models few times to understand the concept again and have all the right functions and hyper-paramters to finally get the model working. \n",
    "\n",
    "I am happy with the efforts on the track #1. Now my next challange is to get this going on track #2. Once I have track #2 solved, then generalize the model such that the same model works on both the tracks.\n",
    "\n",
    "You can find the o/p of the Model o/p of Track #1 in the below links.\n",
    "\n",
    "#1. Car Camera View Video:\n",
    "https://youtu.be/joLZOyeKu1U"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "image/jpeg": "/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEABALDA4MChAODQ4SERATGCgaGBYWGDEjJR0oOjM9PDkz\nODdASFxOQERXRTc4UG1RV19iZ2hnPk1xeXBkeFxlZ2MBERISGBUYLxoaL2NCOEJjY2NjY2NjY2Nj\nY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY//AABEIAWgB4AMBIgACEQED\nEQH/xAAbAAEAAwEBAQEAAAAAAAAAAAAAAgMEAQUGB//EAEoQAAEEAAQCBQkGAwYDBwUAAAEAAgMR\nBBIhMUFRBRMiYeEGFBYyZHGBkaMjQlKhscEz0fAVRGJyg+IkU3M0Q1RjkpOiJYKywtL/xAAZAQEB\nAQEBAQAAAAAAAAAAAAAAAQIDBAX/xAAlEQEAAgIBBQACAgMAAAAAAAAAARECEiEDEzFBUQQUImEy\nQnH/2gAMAwEAAhEDEQA/APz9ERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBER\nAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQER\nEBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAR\nEQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREB\nERAREQEREBERAREQEREBF9OPI+/799H/AHJ6H7f8d9H/AHLemTO0PmEX0/oh7d9L/cnoh7d9LxTt\n5G8PmEX0/oh7d9L/AHLnoj7d9LxTt5fDeHzKL6b0R9u+l4rnol7d9LxTt5fDaHzSL6X0T9t+l4rn\non7b9LxTt5G8Pm0X0nop7b9LxXPRT236XinbyN4fOIvo/RX2z6Xiueivtn0vFO3kbQ+dRfRei3tn\n0vFPRb2z6XinbyNofOovofRf2z6Xinov7Z9LxTt5G8PnkX0Pov7Z9LxXPRj2z6XinbyNofPovoPR\nj2v6Xinoz7X9PxTt5G0Pn0Xv+jPtf0/FPRr2v6finbyNoeAi9/0a9r+n4p6Ne1/T8U7eRtDwEXve\njXtf0/FPRr2v6fimmRtDwUXvejXtf0/FPRr2v6fimmRtDwUXvejftf0/FPRv2v6fimmRtDwUXvej\n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\nweV1f3H63+1TWSJh7bMC86vcAFe3BxM7Rs/5tl847yucfVwYb/qX+y56WjQ+Yku/xT2PlSaytw+o\nEjGjKyz7gSLXftn32A0Ab2D+W6+bj8sWM26OLRybPlHyyrrvLSjcXRzGHiesu/8A4prJs+i6gvjB\nzyOcTs3QfmpebRmOy1tgbkr5X0vlsnzd4vlORX5KA8qQLvBX/q+CuiW+q+zcxrRMQAQaY02ol8bH\nFvbd32F85D5X9V/cGnSv4lH/APFRl8resOmAa0f9TwTVLfS54yNI7Uc5GzWhfMM8qCwV5qT75fBS\n9KvY/q+C1qly+m6534W/JOsd3L5n0r9j+r4J6VexfV8E1Ll9N1ju5d6x3cvmPSr2L6vgnpX7F9Xw\nTVLl9N1juQTO7kF8z6VexfV8E9K/Yvq+CarcvpQ7m1q45rHbxhfN+lXsX1fBd9K/Yvq+CmpcvoXQ\nxkVkrvCr82Zwc8NO68Rvla0A5sASeBE1f/qoy+VrnNd1eDyOOx626/JNS5eycO4OIEgrhnBOnwVP\nVzl5b1eYXoWuAsfNeLgvKSTDsySwGfX78ngVa7yojPq4AtHITf7VdVtvL9DmD213fuuUW3lNV32v\nMf5SFzcow3Y5Okv9lWem4i8OOCLaFVHLkH5BTUt7jMVI00SCPctMeNaRr+a+YHTrs7i7DgtvsjNq\nB761Xf7cr+7fN/gpqtvrGTNcN1PMKO+v9ar5FvT5bth/qeCtHlK4CvNtP+p4KayXD6WSFsjnPkYx\n50oDfZUvwz25G4eYt17QcLXhjyoIr/gx/wC54KXpWf8Awde6XwVqTh7jOuws73tja5mU9r965jgu\nZY5i6R5cxxbTaFa8z3Lw/Sk6kYPU85fBdPlUHNp2Bv8A1fBWpR6pgdHE0teWlzqLALoVuqZ2uirK\nbG45rB6VUOzgq90vgq5vKVkrABgGtcPvB+p9+mqVI9ZjOuhzt9YbhGQhx7Q2Xis8oDH6mGrn9p4L\njvKFznZhh6P+fwSpHsPBYbaVKLE604LxH9PBw/7LR4nrPBRHTfPDX/8Af4K1KPq4XMDT1T6P4OBV\nMpyuBLSw9y+dHlBVf8Ntyk8FoHlTpT8EHjvk8EqR7+RsjQ+7PNUOjLCHC2kcQvEb5TZHEswmUHh1\nngrT5Vh2+B+r4JUj32TOkj9UEt3o6qpkwMhbno8nA0vnR5RFshczC1/qeClJ5RiTU4OncxJ4JUjw\nURFoEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQER\nEBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAR\nEQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREB\nERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQ\nEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBER\nAREQEREBERAREQEREBERB//Z\n",
      "text/html": [
       "\n",
       "        <iframe\n",
       "            width=\"400\"\n",
       "            height=\"300\"\n",
       "            src=\"https://www.youtube.com/embed/joLZOyeKu1U\"\n",
       "            frameborder=\"0\"\n",
       "            allowfullscreen\n",
       "        ></iframe>\n",
       "        "
      ],
      "text/plain": [
       "<IPython.lib.display.YouTubeVideo at 0x110619128>"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from IPython.display import YouTubeVideo\n",
    "YouTubeVideo('joLZOyeKu1U')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#2. Complete Car View Vidoe:\n",
    "https://youtu.be/uEzU61ZdIGA"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "image/jpeg": "/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEABALDA4MChAODQ4SERATGCgaGBYWGDEjJR0oOjM9PDkz\nODdASFxOQERXRTc4UG1RV19iZ2hnPk1xeXBkeFxlZ2MBERISGBUYLxoaL2NCOEJjY2NjY2NjY2Nj\nY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY//AABEIAWgB4AMBIgACEQED\nEQH/xAAbAAABBQEBAAAAAAAAAAAAAAAAAQIDBAUGB//EAEsQAAEDAgMEBQUNBgUEAwEBAAEAAhED\nBBIhMQUTQVEiYXGR0RQWMlSBBhUjUlOSoaKjscHh8CQzQmJy8SU0NZPSQ2NzgkRVg5RF/8QAGQEB\nAQEBAQEAAAAAAAAAAAAAAAECAwQF/8QAJREBAQEBAAICAgMAAgMAAAAAAAERAhIxAyETQSIyUWGR\nBBRC/9oADAMBAAIRAxEAPwDz9CEIBCEIBCEIBCEIBCEIBCEIBCEIBCEIBCEIBCEIBCEIBCEIBCEI\nBCEIBCEIBCEIBCEIBCEIBCEIBCEIBCEIBCEIBCEIBCEIBCEIBCEIBCEIBCEIBCEIBCEIBC3PM/b3\nqH2zP+SPM/b3qP2zP+SYmsNC3PM/bvqP2zP+SPM/b3qP2zP+SuGsNC3PNDbvqP2rP+STzQ276j9q\nz/kmGsRC2/NHbvqP2rPFHmltz1H7VnimGsRC2vNLbnqP2rPFHmntv1L7VnimU2MVC2vNPbfqX2rP\nFJ5q7aH/AML7VnimU2MZC2PNXbXqX2rPFL5q7a9S+1Z4plNjGQtjzV216l9qzxR5q7a9S+1Z4plN\njHQtnzV216l9qzxSeau2vU/tWeKZTYx0LY81dtep/as8Uea22fU/tWeKZTYx0LY81ts+p/as8Uvm\nrtr1L7VnimU2MZC2fNTbXqX2rPFHmptv1L7VnimU2MZC2fNTbXqX2rPFHmptr1L7VnimU2MZC2fN\nXbXqX2rPFHmptr1L7VnimU2MZC2fNTbXqX2rPFHmptr1L7VnimU2MZC2fNXbXqX2rPFHmrtr1L7V\nnimU2MZC2fNTbXqX2rPFHmptv1L7VnimU2MZC2fNTbXqX2rPFJ5rbZ9T+1Z4plNjHQtjzW216n9q\nzxS+au2vUvtWeKZTYxkLZ81Ntepfas8Uvmntv1L7VnimU2MVC2vNPbfqX2rPFJ5qbb9S+1Z4plNj\n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\nhAIQhAIQhAIQhB//2Q==\n",
      "text/html": [
       "\n",
       "        <iframe\n",
       "            width=\"400\"\n",
       "            height=\"300\"\n",
       "            src=\"https://www.youtube.com/embed/uEzU61ZdIGA\"\n",
       "            frameborder=\"0\"\n",
       "            allowfullscreen\n",
       "        ></iframe>\n",
       "        "
      ],
      "text/plain": [
       "<IPython.lib.display.YouTubeVideo at 0x1105f9ac8>"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from IPython.display import YouTubeVideo\n",
    "YouTubeVideo('uEzU61ZdIGA')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Track2 (Challange Track)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Final after lots of efforts I was able to successfuly run the same Model on Track#2 as well. (Thanks to Luc Frachon - another student of Udacity for showing that predicted steering angel from the model can be further calibrated in run time as well. This was the majore single parameter change that made my Model work on Track#2.)\n",
    "\n",
    "So all I had to do was calibrate the steering angel by multplying it to get the correct steering angle so the car stays inside the tracks.\n",
    "\n",
    "Below parameter worked for my model. I multiplied the predicted Steering angle by 2.5 in the drive.py.\n",
    "\n",
    "steering_angle = float(model.predict(image_array[None, :, :, :], batch_size=1))*2.5"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Below are the Vidoe files showing the Model performance on the Track#2.\n",
    "\n",
    "#1. Full Car View:\n",
    "https://youtu.be/D117-ddNcpQ"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "image/jpeg": "/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEABALDA4MChAODQ4SERATGCgaGBYWGDEjJR0oOjM9PDkz\nODdASFxOQERXRTc4UG1RV19iZ2hnPk1xeXBkeFxlZ2MBERISGBUYLxoaL2NCOEJjY2NjY2NjY2Nj\nY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY//AABEIAWgB4AMBIgACEQED\nEQH/xAAbAAABBQEBAAAAAAAAAAAAAAAAAQIDBAUGB//EAEgQAAEDAgMEBQcICQMEAwEAAAEAAhED\nBBIhMQUTQVEUImGR0RZSVHGBkvAVMlOho7HB4QYjJEJVYnKi8SUzkzRDgtI1REVj/8QAGAEBAQEB\nAQAAAAAAAAAAAAAAAAECAwT/xAAiEQEBAQEBAAMAAgIDAAAAAAAAARECEgMhMRNBIjJRYXH/2gAM\nAwEAAhEDEQA/APP0IQgEIQgEIQgEIQgEIQgEIQgEIQgEIQgEIQgEIQgEIQgEIQgEIQgEIQgEIQgE\nIQgEIQgEIQgEIQgEIQgEIQgEIQgEIQgEIQgEIQgEIQgEIQgEIQgEIQgEIQgELc8j9vegfbU//ZHk\nft70H7Zn/smJrDQtzyP276D9sz/2R5H7e9B+2Z/7K4aw0Lc8kNu+g/as/wDZHkht30H7Vn/smGsN\nC2/JHbvoP2rPFHkltz0H7VnimGsRC2vJLbnoP2rPFHkntv0L7VnimU2MVC2vJPbfoX2rPFJ5K7aH\n/wBL7VnimU2MZC2PJXbXof2rPFL5K7a9C+1Z4plNjGQtjyV216F9qzxR5K7a9C+1Z4plNjHQtnyV\n216F9qzxSeSu2vQ/tWeKZTYx0LY8ldteh/as8UeS22fQ/tWeKZTYx0LY8lts+h/as8Uvkrtr0L7V\nnimU2MZC2fJTbXoX2rPFHkptv0L7VnimU2MZC2fJTbXoX2rPFHkrtr0L7VnimU2MZC2fJXbXoX2r\nPFHkptr0L7VnimU2MZC2fJTbXoX2rPFHkptr0L7VnimU2MZC2fJXbXoX2rPFHkrtr0L7VnimU2MZ\nC2fJTbXoX2rPFHkptv0L7VnimU2MZC2fJTbXoX2rPFJ5LbZ9D+1Z4plNjHQtjyW216H9qzxS+Su2\nvQvtWeKZTYxkLZ8lNtehfas8Uvkntv0L7VnimU2MVC2vJPbfoX2rPFJ5Kbb9C+1Z4plNjGQtnyU2\n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\nCEIBCEIBCEIP/9k=\n",
      "text/html": [
       "\n",
       "        <iframe\n",
       "            width=\"400\"\n",
       "            height=\"300\"\n",
       "            src=\"https://www.youtube.com/embed/D117-ddNcpQ\"\n",
       "            frameborder=\"0\"\n",
       "            allowfullscreen\n",
       "        ></iframe>\n",
       "        "
      ],
      "text/plain": [
       "<IPython.lib.display.YouTubeVideo at 0x110619a20>"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from IPython.display import YouTubeVideo\n",
    "YouTubeVideo('D117-ddNcpQ')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#2. Car Camera View Vidoe: \n",
    "https://youtu.be/9IyZikbVlr4"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "image/jpeg": "/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEABALDA4MChAODQ4SERATGCgaGBYWGDEjJR0oOjM9PDkz\nODdASFxOQERXRTc4UG1RV19iZ2hnPk1xeXBkeFxlZ2MBERISGBUYLxoaL2NCOEJjY2NjY2NjY2Nj\nY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY//AABEIAWgB4AMBIgACEQED\nEQH/xAAbAAEAAgMBAQAAAAAAAAAAAAAAAgMBBAUGB//EAEgQAAEEAAQCBQgIBAMGBwEAAAEAAgMR\nBBIhMUFRBRMiYXEGMmSBkaGj4RQVFiNCorHiNFLB0TNi8CRUcnODkzVDRGOCwvFF/8QAGAEBAQEB\nAQAAAAAAAAAAAAAAAAECAwT/xAAcEQEBAQEBAQEBAQAAAAAAAAAAARECEgMhMRP/2gAMAwEAAhED\nEQA/APn6IiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIi\nICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAi\nIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiIC\nIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIg\nIiICLddgA0f4v5fmofQ/8/uU2DVRbRwZ4P8AcoPwxZ+K/UmihFZ1fenVd/uVFaKwx1x9yx1feggi\ntbDmPne5ZdBl/F7kFKKwRWd/csmGvxe5BUis6rv9ydV3+5BWis6rv9yyIr/F7kFSK3qf83uWeo08\n73KaKUVzYAd316luw9FRzA5MVZDcxAjuh7U2DmIu3hfJ8Ymfq/pRa3LecxafqpnycZb6xwpux6vz\nvemwcFF25PJ10eUOxBBdt9381H6h1I+k7f5Pmtzi3+JscZF2PqL0n8nzWW9AFx/iNLq+r+alln9N\nlcZF3Ps4cwAxO/8A7fzVMnQvV3/tF0L8z5rHqNY5KLqxdDCRt/SK/wDh81M9BUP4n4fzW5zb/Etx\nx0XY+ovSfyfNPqL0n8nzV8dM+o46LsfUXpP5Pmn1F6T+T5p46PUcdF2PqL0n8nzT6i9J/J808dL6\njjoux9Rek/k+afUXpP5Pmnjo9Rx0XY+ovSfyfNPqL0n8nzTx0eo46LsfUXpP5Pmn1F6T+T5p46T1\nHHRdn6h9J/J80+ofSfyfNPHR6jjIux9Q+k/k+afUPpP5Pmnjo9Rx0XZ+ofSfyfNPqH0n8nzTx0eo\n4yLs/UPpPw/mn1D6T8P5p46X1HGRdn6h9J+H80+ofSfh/NPHR6jjIuz9Qek/D+az9Qek/D+aeOk9\nRxUXaHQF3/tOv/L+ask8mXxtY52J0eLHY+aeb/F9RwUXa+z/AKT8P5rH1B6V+T5p4qeo4yLtfUHp\nXw/mss8ni8EtxJNf+381LLP6uxEjuUaFLEGYNJDuwda21Ush8TS46rFquQZmFWDx1URoCrKNLZFl\n47ZWFsCnBE4IMsNFSkNhV8Vk7KDLNwsv3UFY5zTGBWvNUQREQFkbrCyBRQZUuCgVIHRQC7LQA3W1\ngcY7C9YaBa8U4Fa1Zq5LJtooFQegw2Ple1oiIIOmUAI6btU0ZeJbuuLhsSYPONtzWW8108O9ksnW\nPdehJPqWajZE/WyEOB40Sp6XvRKlD0g5rY2Zj2QAdNAFs4jFRyglwY5tdltaDvWuPtefwsaYJdTA\nNStlsJjh13rVVxYtgmdla3Ll5Vqro5opg1spNlp04Eqd/S9EiMJJ7Z8GrSkhfIS5o3/RdCUAEkuG\n1NA4qTgLaRwXNtw39ZhzYGgWzBI6WJr3UFvSMbm02K5UgySOkZoQaBXf5/TGOo3EUY5GPFtPqUl6\npZXKzGEWaRUYRZRQYSlmlmkEaWVmkpBhFmkpBikUqSkEVlZpKQYRZpKQEWaSlRhFmkpQYWVlrbPf\nwW3HgxJHq/tcgs9fScEh0dBE+UOe1zwNQBw8VHpbFZ5msoNDdNEEWIAe2OPsXsSue5gE7zIO03Ve\nK9+/p6dfLdwcbJJgyV1N3JWekC36WDCWujrgtIudI4ZO0TwU3HqQM+nClufTe0vI81lDSLcaC2WH\nq3ZRw5cVqxMdNKHnYEAFS6QnGGykDtngr9utv4vMedY8b1Y4Wdla3EDq8tWfFVwYd8gFisx0WOrc\n2QsO7SQsWKsDuNapai85RfEo3RBCSOwSteq3W0DYdfBUyt7VrUor4oiwVoL1UgLVfFWsQR40s32a\n4JXbWCgDVFhZQE4os1qgWpV2VHipAjLSCxgoC1CQ9rRHu1FKG5UEgLCugc6NzS1xHdzWuCrmu08E\no7EeNjMDw9oY/TzReb+ysqKSMSRim3Xj6lxo7k2JtbVllUSANuSxg3KLS4HKR42oZ6uqrh3eCjHO\nXOGc6EcANkkML2Dqc3geXBTBL6Q5wHbO63WYzrHbUfFczzR3qTX5XXsmK6LJyWl96i1pSDrHBo4n\nMT/RTjcWxHXcIARDmO7jomq13Esf2DQHFbeFnE1tJAeOZVYhzxaDUla74pI3ZmkBw2W+PpYxY6g1\nCUqoZw6EGQhr+IVjHB1061653K54epZCsbG4tLqNDisuhIvttOVt0EvUMVcUWdANNUWvxGEpZRBi\nlmllYRNERKQEWURWEWUVQRZpKUVhFlFRRiZHR1lvXkrsPLM1gOYgrDow8i+CkaFVwXC/P11+rKql\nxk7JAGSuscMyvYI5w6SWRoeW6DmVzZ2vdLZNOdsAtgR9U2MX94SCvL439dZUXdbCzrICTvsqDM+c\nkyAk8LK2YwSXNB15KlrDZcBos4rbdiQzCRuDctAt35cVzJpXTy5nONcibVsziWCPg0laxGXZBtYc\nhuHdIDZdstM4d5cXNaKA1U4C/wCjUbaA7TuC2YCA0nhXtW0cl4JOvsU6quatkH3p09yrvWlBAjc6\n7o2Pri1l5dd1J7miMChZJsqEZIGlg3vasFErHQvyyNpwUd12elcL1sQlbVgarlMZdmtgt6KuKtZs\nVCt91IGhoqMv3VakXWsIMLKwsoJGqQUN1GuSk5KMmi40rmxg5RtW6oYe2FaJaDtKtQQew5zlFhYa\n3e91cw6Et0WHtJHIoKANaVgFGlhgyu1V0bcz83AIJx01wIVo7WqiGtLQRzCOfTjlFUFBiR1uDQrm\nf4YB3Cpaw6OdxU7IBJUFgJqipGt+SgDqOarc/tOA5qDba+9qqluBrHxtYHAUFzmCnDuFroYCPPKx\njm+ffqUqrJGGJobevILWlJdTiKI3pdNsHWk6C7oKJwUYneJmksa3Nposq45BLqI1Khoa9gXb+hwG\n2Vl7OYO4ju965ZjGUuA0DqTazYxHM+Jwdd9ynLiXSsPmtHcrX4AnI0P84XstkdDCCMPfKCDsMlf1\nT9MakMw7Rlc4ngasLcbjIjA+KK2k+cTxUZcDlgdKwkAGi0aAKH0NuWPKKzNc7Tupatpiz6W2a2SM\ntzSCX91UlsncXMpgLqytC0piGiPKdXA5vaq4sQ6EBzCbvg6leerGfLpCJ+ctyG7oHvWOqkrRrr7w\ntWLpGW6tzjYok7Lah6TnYS2ThxJ24rp/tYnkykAWDvVrFUSgmfK8RC8pO5U5Zo7bExmlmncTfct8\n/ff6l5QRVvxMYcQbBG+imx7JBbXBd53KzjKLI18OaUtIIsogwlKVaLG2hQAlA6LBcBuQPHRa82Pi\nYSA6yOIGizepFkbmJgjY5jgSQW0dOKy7DM62OQHstYHO/qtbo7E/TJYo5Mwbn1cbdrS2cfEcDiJI\nqOWVxI5NF6j3rw9fS66RrYamykji7jyU3RZsQ+yQ1x7SGFzmdZH2WjmokOkBDnkcTSw21sREXPPV\nt0Gm/BUTYOdrh934GxstpwvieW6m2Vxj7duyivUg55zERtc3KQNQrMtaHYcFc7Uaauduq+rJNADk\ntCE8cZjEjmnT8Q4LQcRWnErpv+/+4F9W3chaWLjbG9rW5coboRxCCvCODZ3F2XUENzc+CpcXPkcT\nWYHhspM0e0jmCByV+FaRiMzxWU5j4KjqZQ6JzHag7DmuPI0R52GwTstpri5pc4arYYzLlHGrKmjk\ntwk+JLRDGXlUvieyw5pFL1MTRHE6U3touQR1kln8S1o5Cyt/GYEw2YxoRa0boF2/Ba1EUCaZq2Cl\n4hUBusjUrB7JRuiCQYc3cstF2OKnC/mFa1jS5xCyKwKbooEuz6nRXdU7hqqzeYiqQS6sPFN1duro\nGZWG9yq4z1Z0VrJBl19SUTa0Mo8FlwHV3tmKiHW2uKy/7yNovZZEstNaX7EUFItDi3kFgakNPBYL\nqdaCXV/eOdejQqWNsN5uKvacwe5+1IDTWk7AIqNmyTwFLsYCVgkZM/QNFLlRW9paRodVtxkFjIm7\nndKN4zuDW1oSBr61kzvLHni53V34aKmJ9vaDxLj+VQg1ljbwHaUVuZ805bwDcpPsXOaP9nA5yk+4\nLpQx5g5/4rWrla3DsaL3UF7Xhs2Hc4W1up8NFsY2TNJAxruzep5Ba2Hysne59loblpQlAb1pDrJ0\nYO+lBiaZ0eDjbeknnKbJNz/JhXe3KqMUQWxRjgViYZDIOJYG+5UaOIDmUDuBft1VJ4BbfSOs58G/\noFqHzh4KxBjq9oXQL2u60HckfouczUetXs1kI7kqO7hsG1ssBdM0mSiGj9CnSjYfrF8cEeR0dX31\nxWhC6SPGRuOgaV2+jMFHicdis+YPmYSwjYXe/uWFeaxRDi4kcVqmQtfbSR/Vb+OidDJI2qyOy+K5\n7gt89WJY6uGxMczB2qdWoPBX9/NefYac4EVfFd7BtmOGjLgHZmm+BaL4r1cfXP653lKlkD+6zK9k\nLAXOAC05+kGN0jBdY0K6X6xPLYnmELczvUtJ+Mkfly00nlqqJZ5MS4CQVl2pVxvGerXn7+tv8anK\n5gc9xLiSD3rUnGSavbyW+3cVtapxUREgP8y896t/rcjY6MZ9zK9mmUg7+C3ek8a3HNgppa6NoaXH\nidFt+T2DBwOLkrQDT3LjG34tzBtmKk/Wo3OtMjWxcAFaHRRsfENXVutTP1LieIWqZsxOu+i1ithh\nzUpupkMl7nZa+H0e8E8FKckRjvUSs7O71CeVsURA1e7TwWJ5hEwuK08MevxAe42Gm/7LdR0sLDkj\n13dutXHYRjA6ZpAAq216l0o7MRPBa3SADsFKS4C6rv1Cg5WEi67EBuag02tmbL1prY71xUsHAYYj\nM8auHZpQecu+nFFSiZmfoeyNSt3CRPmla1ovMaWvEwthHN+q9J0VhW4XomXHv1c4ZWDkdVByuknC\nGLqg66NLlQjNNXALaxz80jtbI1PitbDblWDZA68PHBvMqMGFwMjxBi4W9m+205bvmrmjLhxQ1J25\nqmTCPMmfOBY2PFXUSxnkxYe/CTtGxbE5te+/6LmT9D4+CTL9HfLbQbiaXDwul6PA4/qvu5XE8Ad6\nW/BiYpycrwKFLWj5+5hz5HW1w0II1ClKO1ovd4jB4fEsEc8DXDNdWQT7FzZvJzCPkuOV7BXZadv7\nq6PMtGQEkaAKOcijdX7l1sX0FjYSymsma7cxuPvuly54nQPdE8ZXjcFETZIQBqpGtSdVq6t4oJHF\n26mDablc3ZTAzjs7DitRzyb4BWxEsaMp0QW6tJtSjcM2nBUBxdIb2pXRhoYOZUFsWrm3uVPR0+Qa\ngHdRYKcCFIVE0k6k2oDm2SPwqJF0OAVUT3Pc4E6WrGnKC5BY0loJ4BWwvrrJXaUOyotAe0Xpm2Rw\npzW8CUVv6wvt4rI2jemp0VkMTTH1sUgc9rSC3kox4lj5ZCWgmQ2b4JJixFFK6NraFH9Vgbf0iIN6\ng2CeIWHluW2NAC5+DxIxEDi5otrst3z1WzKSzPro0Ae9P4qEM7nNs8/aq4QZMUNdAcx71XFdkg6D\nRSwrqkkdyVRJjesxETtxV/qrcQxzp2ka2T7lTCZHEFg1DdFt5gwNed8h9tIrmYslziSFrSaEeC3p\no3usnZacrbZYViK2HsrZw38Rz7lrxjQ2NbXQwcdklupsD3JR3cZg442QOYAWvfRI4Hkrei8ZFB0t\nMXS1ExpG9DRa2MkP0CIN0JmLwL2GWlw5J3GSStiTqsDZ6TnGKxEjxVFxqlz3MBvgVB8pvRGuJOqI\nMBjkEhyEDXK7ZbeJx5ZGQ133j9CW6cN1pSmyG0qZj2wOK0NkOMgzOJc48zarlGXxUoeyRapxDyST\nwVgnAfu3HieKxGfvQsQmoqRh+8Co6DeCljN41rslBkAVuLd2mLFV6Xot3UdAyEHWRx9WgXmYXViX\nO/zH9V2cLOXdHiPhS4Ug6ueQD8LiFOYqWLlt5AKoi0cLUXOJcSpA5Wa7ldRuxhpJd3LPSMjLbGwg\n0LtUQStZ1mcn/DIb40tN8ljN+LvUFU8hxElnYLbwbMjfFacTC54vZb8H+NG3vCI68sXV4eNp33Pj\nw9y0JD12IEY8yP8AVb/SMnVYQSOPadsLWhh25IwDq92pKgum7QDeDRZXOcOukDG/iOvgt7Eu6qIg\nnVw37lrYFmpkOn9EVsNbcrWN4u07gu90riQ3BYbDN0DGaj1BcbAxmXGMA23K2Ok5s8ru7sg9wUHH\nxjjXeTashaWRC1VlOInonshboGoB2CCcTL1OyjNIA7SqGwUZJw0ZWqjzu9ME2B0jrO5K3eko8PD0\nfH5wnI0rjtuq8KwAh2wbqVr4mV2ImMjrDGHsi1YjXjfjYY3iCcsbWY7LEfS2Nwjx1kglYavONhfC\nuKjLNplB84rXmAex2mvLvW4PYwzdfBHKGkNe0OF96m7K7QrW8n8U3F9FRxPLOtiGShyGgv2Lbkjc\nx2rdOfBUcrEdB4OVgAZko/8Al0CufP5OgPuObK0fzCyvR7IaIpw0QePm6LxELSTHmaDuCNVrHKGN\no6gcuK9lJASbZVcloS4ONodlgiBIrtRCvUoPNxm2OdSxCa1K3ZcIIhlyEDmqOraLCiJRPz9nmpzE\n9YGBpOnAJEGMINUtqOQMN0HLKtGItF8COauYM0eo0zLZlbG0ZnMZrvbVJoaHnQXv3KivUdWQOKjL\nKIpbPAKvO4tfITq0nZVZ+tlDn6jkpUdLD4uFzHHqzm46Kp8kcjxF1Tbc6tRzV8Aw3Udh9SO1LQKA\n9a1cZKyORhYGl24I3J7isjZwmGGEzt6wOt2bQVl7kkmc9tvZVu1ykLVlxbhEMun8wBu1SJw5tPc4\nOu2kcUu1W8Zmua2PrAxhPBpsALWDXNzuY/mO8hVucdCCQSDvsQsMfmcWkGx+K90R28K5scJsgF2q\nxKA+duugqytGNugJsWaHetmycNI1gt11YRVrnNZim1RaRqqHwtdII3eaTehUG3nIcNQNbV0DSXNq\njRvUKqrfhGZzZO9HupbWFl+rjnw7+1/mNqjFSC3PsEF50HHVa0kjZGktYGjZQbmJxbpI2GV+Z2xv\nktGfEwCFzY2O6w6Ncq5CtWUnSj7ERllgWTZVjSQ1Vj1+1ZL+0AqiY1NoyMySZq0CwTTb9i2AOqgs\n8UVS0GQuP4RoqsQzKxbcTMsVKvEt7I0VVrM0AWTo4LNUAsHdVEo3VIFfM/UErVBp4Vj3WNVKj0vk\n9GZ3AnRrLLjy0XH6Rr6bO5vmue5w9ZW10ZjThOj5mN0Mp1PsVOIgfODIwWALJAWYrmWTIrgzPtwV\nbWua9dJsbYsMLHaJslb0c6YVt4KtjHTaDnyWzI0HNm5ldKFsMeAHVsAcT51IrkjsNrS1s9GMzYlr\njprx58FqVmVzX9XEcpouFBEbvSGIONxuRt9RBo2wByu677U8MwvkLiNBoFThosrMm3Ercefo2EJb\nudFBzsc7rcQWja8qvAEULRwrVVQR3JmPNZxT9QwIrp9EsywT4g7+a3+v6rnY2QFxAK60hGG6LjYD\nv2/Xt/RcFxzyFzuJ1Kguw7Mrc3NYfNvSqkmumt2UCbNKoyCXHMVIGtlFxytpZhFkWVVdGNp+iF38\nxpamM+6jaznwXRbJlgDzszZcTFy9ZKXniSoNVxOcqya2Za4auRoAeHFJi5z6A3/RVHQ8m5Wx4rER\ndrNJ2m1tpf8Adeww4biGZH+dXZJXguj3yQ9LYXq7GZ4YQOIsL2ubqznboQd/6LQxPGY5MpCrXUe1\nuNw+YUJQueY6JBsEcEVXXJKB84WO9SI7kyoNeXBQTNNCiea5WK6FksuiojxXcy0sg0oY8c+KSJ1S\nNojgVT1r+tyWKpe0mihxTcsrLXMm8m4nOfJhpSHHYGqCYy5c+KilytbpQo3zUnGm5hvWySdD4nDP\nJfGXAcRRVD5T9ZMOWgGHca8UxSL+HeTxslajtJ21sVdC9z3PadnWQEGFznsHW9VBbA8ukIJ7AHqV\nbonSyvcwHsLoYbBPEvV5PP7WvIf/AIrA0lzomiwXHc8+KxUc7DxODzM/8OobzWuGktyu1LPNI4Bd\nKRohdFV5mEjT2rVngbHL1THEhwu/6K6qzCsZKGNe0uaRr4qudnVPLLoDbmFdBmgc1sorQEcd1nEN\nzl2vZVF7ZGfQsrgQ4G2n2K3BYgmFwa0dhxvvXKBe4hvI0ul0e3K14vjRUwXsfHM3ETOFO009S1pc\nUGRDq9HEKoS115B0Jpa0THPljbzdZVab+Hw8mIw7q1Jc33rc6UwsOFihihNmrd4rZ6OBic4jgCfY\ntLpJ2fFE8RX6LKOZKd1qnUq6Y6lUgceSsZS7lEAmyAsgXZ5q7JoG8dyqKBbpo28jZW1JJncG8AqI\n2XK93JXRNoFx4orYZqKUJm2a5K7DMzHXYaqBp8kx5OKitSQVQVZFLYLezqFU9hEdq6lUndSUXCgL\nVjRZCI3I9Ia9i7vQxa3A4lzwCTGWm+BIK882Ux5S0iwb+a6jMcZo87qzPAz1xPNYqqosG0yukeLY\nDp3rGIsknmVtPcBhIrO5J95VYYHts6AkJqtDpCEsiY9uuYKfR8p6t0T/AOXM3xW9iomufwMd1a1p\nYxHNlbtS0rnjRpWYGGaYD8LNSozSsZGWtGq3ujIaY2x52pVRtMbQ21KjiT1kojHmtHtV7nBpLuS0\n4/NLzuUGCRGy+S1cMDisaBsPas42U1TVf0K370v4gKDb6YnzEMaANNh4riuJDqW1O92IxGYXqjMM\nXG3HbdUa7RTcxRjdSSrJqL8jdgq3PHqCA43otnDRkkWqIYy8g1a3XPEbQ1u6DGLm7ORuy5cpBdS3\nHm9XLVfl60WKCCyPD52A7BWCNtUtiNo6oVsULAXbexTUczFh8TmysOVzDmBBoil7Ho+duPwkc8TW\n6ii0OzZT3+O68xjIwWOAGlFeg8j8M9nRHWEjLLIS2uFWNfYtSjtYXNG4ECjyVs2GMpL67XILbwuD\nFB7xosYrERxdmIarQ5L4nNOoUKpbbnmTelWYwitci1AiuC2TGomNBr0pDSqv2qwxKOSkGQ8nQ1XG\n+KoxHR2FxNkxNzHiBRV226C+CI4k/k6YXOkgdV7NIWqzAzQjtt1vgvUsmc0jMA4d6llhnslmVTB5\n102Jw8uaSz2coJ4DVaYkIsVre69PP0VHM3RxK5GI6FmZZjIIWLByXklxzEmxVgo2ngAgWDt/buV0\nmEniJL2UAqQ0ggg04cVkWPPYaeAKrkOrg3iNlJ7w8XVHY9yhGQZDfALcFcRqTbZX4aQtMzb31/Va\nrbEpKlC/756qsi+pffE7LsYTBsblvca3S5sEfWxbaB9fquyJBHE554FQbDHNikcDtlK4WJmzYh5v\nit9kv0gTScGtXGce0e8rIhKbdShdArPnOscAsHh3qsrIxYHJTYcz3O4KJ7DKWR2WBnElUWNAbEf8\nywfNaFCeSpGRN4bqxnbcGtVVNuJEGHf/ADFRw0oODmedyd/G1RiWGyOAVbXlsOQHQlSq6UcYduNg\nqcRXV6BWxSHKBxI1VeOaY4gOayjUlFhvepVVBRYC7Lam8HPoqiEh1G62cKdhfFabyb1VsJcNRtSh\nHRxb3GJhums08VudGvEkJElFuvq0VWIgD+iIizfUn3rRgxskMUkTDla/ztAVGmevew5S4lt8FdjX\n5JtbHZF+wKGFhBa6aTWtQOZUMY4yuLnbkUtQUtweR7XXfcuthiGM1HqXHdiJOsA2rvXSw8ueLrLv\nL+qolipLAiG5OqomeGtpMwdIXuOnBaeKlPPdEUSO6yW+PBdKM5MHk1Dnb+H91z8NEXOur2K6cYy6\nu1KCWHjbGy3blJ3ZgWt0UXyWQOanJGG4cve4Av8ANvRQc593lAvwWzhcA6WQNqyeAWIQCaYAX8+K\n9AIfq/DOp14l27qotHdy4orlPgMJMbRWXc2qQxu6liJhZYw2TuQVSZQ0AAi0EZDrS1n6k2trIdXV\nY5qiXQ7bqhh5Azjot9gGrr0vVc2KGSZ2WKN7zyDSV63oHyemZmkxkbXEGmtcLb791cRo9HdCfT3u\nkxGdkLhTcmhd3i17PA4KLDQsa2MNa3Zo2CsjgiwsYPZBA000A5DkFp4vpAHssPsK1gtxuMa0FrHW\nuVZJLid1hxtZFAIJCllYzDgsFyiwKibWbKwUGLKIiKwWgrFAKRUCqgWhAAFglYQXB5GtqXWHY6hU\nhyzaCbsr9HDRa0vR0Mv4dVZn15KYcpg48/QZjc4x3qDypc53RmIhcHNGf3UvWtlrfXxUs0Z85jT6\nkweEkY6MkOjcHD1qiM08ngV752CwUp1aNQtLEeTuEm/wy1hOugpMHnsC5rYheutranc57Y4Wi3OJ\nJ9i2pPJ2aI/dSZhfJZmwU8c+fI4AnTTZSwRw+GMeEkbXnxhw9hXAk0e7xXqsTK3DwGu07qqocNF5\nSY5nOdsSs4iAFNJ5oNXXy0Q8G9yNCIyBblPXNrwSNuqkxuaQBFargesLiFt4C87id1XiAA4gaUre\nj9yVoTxUeZjua0G6uy967TmNDZCdq4rjf+b61KrsYLDl8QedhxVfSjbcBwGyvwmJrBmKtuK1cVJm\neAeCxUVMZWnIKJaC4grLX3ZVLy7N3Ijfg6ExGN7UFPq/xAcFs4PyfxUjWyvyshvU5hdcTuudhcTP\nh3l0Mj4/+EkLcw/STgQ2bM9gOozfoqR2cFEJZpYzWRoyitAdl5bFRdViXNYeyDVr0E3S+HOHMGDZ\nkc42XE27ZcWeN27qdairMLBNPA5zW5mx6+ACjlfIw6E5RfqXS6Hl6uOaDYyCrPr096pieYBixlDs\n8bmafh1H9kHNiY0O7R1214LpThkGFja1wJcNa58FyM6yHlbGxM+mUN+C1WRvxE1/hClnPJOsIQdC\nKNrG6BSJF6rQ60nilk8VBtfSI4+0W5jwaOKqxOJdiHBzjoBo0cAqnMNKBa7kiNzA41uEmEgjD3Dm\nVsT9KyYkOLtC7iuRRHBC6t9FFbsbwASTussYx8hs1S0WB7zUbXPdvTRa7vRPk7jcYQ+bNEzkas+H\nL1qyDWkktnVwguJoUBuuj0V5MzYsZ8V922/NB1rvXqejug8Pg2guAc8CiTuVtz4uOBtCiRpS3IKM\nD0XhOj4Q0MaNN7JUpuko4wWR6kceC0cRipJzqabyWrSovnxUkrzmf6lQlIipWhk4EKvKDuFMMYNR\nuoM33JYKiSTos5qQSUTusZlEuQZJS9FHP2rIWPPKqpErFo80dN1CxyRErWCsLGYt1CgE0d1aHGlr\ncVc260VEnDisXSlSg4UglmUrCrCyip3SyJCNiq7WEZbLMQ/i73KwYhpPaZr3arTWA6kVvEYWb/Eb\noe6lTN0Rgp2ZRGPEEqsPU2vUxGm/yUwZsxuIcdrJWo7yUkbeWW3Db/VLuiTXevBWiZwGwITB5GTy\nex0bjQYfWf7KEOAnw5PXRGxxXsuuadxSu6kPZ907TkUwfOcXC4yudRHcVHBB3WZANV9BkwdinR3e\n61JOi8MXl3VgO7gmK8v0jceHqtyuNs9e3xHQ0U4DXSvHd/oLTn8m2k9l9gDRSxHCglpmirlJLiV1\n5Ogp2N+7GY8rC1n9HYgE3ER3ghYsHPb5uqPrKtiTDvYcpa6hvoqdh2dDaYjOEj62TLoAdBa6bPJ/\nHOa4xxNcG63Z2XHkBY7MTqNVYMdK5paXmjug3ndB4hrBL1sbb81tm79iwQ+EOgmyl27SFCHPJRDd\nCFKW7t9hRW1g4Jn4aTFs06hzbHdr/Za4xLopHuj2kBDiQOa3+jekY8Ph5MO0diYU/nWv91oY/DRx\nzSHD5nQ1YvhoL96DlgWrGtCgFMXStqMkDkq3BZcTabpogBRUws5VI6BBlstbgrLpb2ruUC90jxHG\nztHZdLA9A4/Fup0YjFi8x1o8lcHMJcSACNV1+jvJvE4xzXzdhm9ZLJ/svTdF+S2GwpD5AHvrUkn9\nF2iYsLGNK/VWRXP6M8nsJgmgtiZn51bvbuui+SGFlFzWN5BaU3SjS3LFd7WubNM5xJcS7uWhu4nH\nuld2LaOQK1C8l2ptUtLjwUwCeNKqnaxqpCmjXVRMoaorBWFF0llYBQWLGw0UbJWWh167IhR4p71J\n5AChdblAN8lEApY5oCOVopXAoLaskihZ1UXqjBtQFkrBvipNcaRGbINELBcbWbHJZJFaIKrynVWt\ncK0KpN77rLXWURbZIWCeaxmpYdrsipB4tZtVkgDZZDxSCVrKgNVKwiGZLWFgHS0E1kFR13RBaJFM\nSrVvvU2uQbOa+KmyRzXdl5A5LXDgdlIOQdKPHECn6jmr8sOIbbJG2uQHKTHFurSR4IremgcwcHDm\nFRR7/WofSJC09pQMruKItdYHP1Ks0Tq1vsWRNe6zmaVMVW6GJ+hYPYqH9E4V2ojjvnkC2qCapg5M\n/QUDvwn1aLRl8nRmuI5AObbXpQ9ZsOUxMeX+q8TA0COUurupa82ExTrzA6d69cYmO8VU/DB3C0xX\njA2eKTWB5A3OU6LZjx7I4XxuBIfoQTS9I/CAXbRS1pMHGDpG31aqYPKhtoXUrNGtVEmp0WcZC9Rv\nQHgVv4PobG4wZo4iG8C4EX4L0nR3kdGMpxDg/Wwcteoi9VryPK4XCT4p1QROeTtwHtXc6P8AJPFY\njt4jNEP5QRa9phej8NhI8scTGj/hAClPjIYG+cCeQK15Vo4LyeweDaC1va4nTVbj5oMJHlOmmwXO\nxHSckmjOyO4rTdJnNucXFMVvzdJOdpEKHMrQe57zqViyUtVUQ0g6qYCxakEQAWCVhzgFW55OwpFS\ndJSgXZtSo8dUuzTQoJDtGgFaGADUqDWlS40CqiWgUS8DxQ6C3GlAkIJFxKZuBUC7vS2goJaBYtYz\n3d+pRDr4KCfeVWXrLzoqjqVRklxUlFviskgDfXkgkCsuGijeik4WxBUWBp5rLa2pYLjwWbLa5lBZ\nlNJslmhqsjXdBW/MViiaCm+uSiDSIDRTFKKzsgxmoVayOSxQJ4LJGiDLXfhKk9oIoKq+SyCgzl7l\ni60pM3es67hAD6KmHKp3vWQ48kVeHKQPeqA4qQKC8FDqqgVIFEZIWA5wUgVhFSE3AqwHiCtcgc0B\nIQbJKiSq2v5lTD2k0oM5lIPpRAtPDVBaHrDg13BV7JmVHnsD5O4zFBolqNpJzUddtK05r0nR/kxh\n8MWPewGQNALrJvnpsu61scLOQ71rzdIxRaN7RTEWx4eKAdlrWqubHwwjTXwXLxGOkmO9BaputSg3\ncR0hJNeUkN4LSe5zjqbKAngsWVVYpZ2RDfBRQnmViuSyAALcfWgcTtoiMhChoCyqy8E6bIMuIG6g\nSTspCqt3sWBrs3RQRpSY4N0bq5HMod5UWtLf7oq0uDR2tygJq2j+yrceQtyiCb13VRI249pZJ4N1\nWC7nQUXGzY0tBF2h71lrlHzjTb7yUblG7h60EgC8kArJGRtFR65oO59QWHyZvNFhBJx1pR2eOSiX\nPOoaAo5XnU0gmCOt7lPQ65VTkdzPqWcprcoLWC3baKReAa4KgAjZxClZ/nKIyc29aJI/MwXuFi3f\nzGlgl3O0VNoNAgqdEO30VNnmsh515ILHdnXgo0TrYpYbIfUlgg9qkGQ7VZvvULAWSc2tojOp4aLP\nrUCdKKNcHbqidcK3UeOVSOo86wFEGzsoGU+pZFjwRxG9G0BBHaQZoHVqwsHQ2NuSNdaCSApSwgnm\nWQbVVqQcqLg5T32VIKmH0oJ2BoRqs5L2UQb3WcwugUVgtybi1dDE2YEA+pVh/AoWBxtpooLzh5We\nZqOXFRDhseyrIOkHxdiYZ2hb4OGxjdSHHv3Qc1w20UCFvS9HllmFx8CtVwLD940tUGJ8XJI3K5x9\nblrZjx1USdde0pgE+CoxmvgrMmlk0gAbwWHedrqEGLJBrgsBpP8ArRWNIA1CrklA21/RAym6HtWQ\n3vsqvO6rugs9YGjRBktAPbdXcsGRrdGNsqA7Rt5WC9tU0etBhz3PPaPqU2ZQLIFeKrYztacd1sCL\n7rO5ttByqorAzuuiRzpWNIa05d1h0kQYHOc3eg29VQ/EROrJnc7W9BXqUFhJc7s6rFnjoO9VdbIW\n9kBvisuc90YYZOzudBugue7JGHBpymxmPcqDK0nV7Qe8qJAJaCXU3cWplzdgK9SKzTXNFZusJIot\n203R/VsjaX00t843eb1cFW7EBu5orXfiI7rVyDJxEk5LYGljB+I8VNkLWDUk/wDEbWucQ47V6lHr\nnu/EiN3st4gLHWMH4gtLMTuULgOKDa+kNCx9IHJavXM/mCiZ2N4hFbn0jTY+1Y6//VrT+kt4Upsl\nMhpjST4IjYMx5H2oJXHfT1qAZPf+C/3LGSYWTE8DwCC3re/3p1trU65qyJm80G111KQlC1esbzWR\nK1BtdaKTODxC17FXSAjmoNm7WQ6hotcE81LO4Ki9rg53a0RoPA6clTnvcLLX15pQbLWl2zdhZpYv\nQ6cLUGTuGx1qlNs4NNLb50gi0kHez7Uzgk6C1J8rOoymM5s2rgoPMUZZG6RoLzTRerlRnNxIKEcl\nZLGWkggbaEFHsMTi12j+LUEQSpCj3KJFGhopCxrSgFtjRV5iCQApmTksOFi+JVUa4uVgNqpwruWG\nuIRGzeiWOSqD1nMgsBUwVUCpAqCwusbetRBLDbbB7kGqyL8UG3h+k5IyGy9pviulHJh8Wyuye4rh\nlgPcVgOfGbBJ8EGa4u25KQrShS8oPLQXZ6Ps/wDO/asny10/8P8AjftVyrr1dqDpABYFleW+2lj/\nAMP+N+1Vnyvs/wAD8b9qZTXqc1m3GzwCwbPnadwXl/th6D8X9qz9sPQPjftTKa9KeyNFir7RHsXm\nW+V1HtYHN/1f2qR8sNCG4AAHnLf/ANUyo9IWhzKdnzXxIpXOMbGZXFhF+cAV5E+VhdWbBk0duu0/\nRY+1Wn8F8X5Jg9a7ENytY0ktFkabLXe5r71ocl5j7Uk2DhDVaVLRHuQeVAu/oVmt+t19tK4PRRsY\nW5h7aUswHG15s+VJO2Erwl+Sj9pnf7r8T5LOUem6wDiq3TNHFecPlIT/AOl+J8lA+UFn+F+J8kyj\n0LsRQ01VLp5Hb+5cQ+UF74UV/wAfyUW9P9p2fDEsI0a2Sq9yZR2TVG9FgEFwDGvcTpo017Vyh0/C\n1uX6AHjnJJnPtIVzPKgNbl+hdkbBstV7lcHVZh8Q6QhzWxCrzOId+hVrcBIXgyT03iI21+q5I8rG\ncejgf+r+1T+2PoHxf2q4jsDo2IOzv6x1cHuCvbgob/h2DlxXmMT5VPxMZacKGk7ESbe5Twvla6CB\nkb8IZC38Rl39yuD1LIWxNyxta0eCnlPNeZPllf8A/P8AjftWPtl6D8b9qJ+vTZTzTKeZXmftl6B8\nb9qfbH0D437U/D9emy95TL/mK8z9sfQPjftT7Y+g/G/ar+H69NR5pRXmPth6D8b9qfbD0D437VPw\n/Xp8rv8AKq3YWJxzOjBcV5z7Y+gD/u/tT7Y+gfF/amRXoHYOCiDGB4KsYCLYSSAH/XJcI+V5JFYK\nuf3vyWfthp/An/vftTIO39Ae0nI8Zdhm1VZw+Jbp1YNcbC4w8rzxwQP/AFfkpfbA/wC5fF+Sz5HU\nJcNC03xNUgLT61yz5YZhT+j2uHLrP2ql/lNE4iujsvcJv2qYrt8dCpdped+0dbYWhw+8+SmPKbSj\nhL/6nyVyj0AkrdSsEjna84fKW98H8T5J9pfRPi/JMo9CxjY5nTMY0PdpmClDmhDs80kumnWURfqA\nXnB5TV/6T4nyWftPrrhLHLrdP0TKPSMxEjGtDoBiJHAjs9kN5blTM0bMQ6Av+9AFgMOml+dsvM/a\nfW/ofxfksO8pw5rmnB6OFOHW7jlsmD1RADM+bMQ6qGvvWC8ncUdwV5X7StELYo8GYmB2aopclnvo\nLMPlVKxx6zDB7ODc9H1mtUwepItt7EqF0fNpee+1lijgvD73b3IfK2xRwXxfkmD0WUXYKnsvL/ao\n/wC5/F+SkPKyh/BfF+So9OCpWvL/AGt9B+L8lkeV3oXxfkmD1IKkHLyv2v8AQfi/tT7Yeg/G/apg\n9ZnWMy8p9r/Qfi/tT7Yeg/F/amDzCIi0giIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIg\nIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIi\nAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIi\nICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAi\nIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiIC\nIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiIP/Z\n",
      "text/html": [
       "\n",
       "        <iframe\n",
       "            width=\"400\"\n",
       "            height=\"300\"\n",
       "            src=\"https://www.youtube.com/embed/9IyZikbVlr4\"\n",
       "            frameborder=\"0\"\n",
       "            allowfullscreen\n",
       "        ></iframe>\n",
       "        "
      ],
      "text/plain": [
       "<IPython.lib.display.YouTubeVideo at 0x1106195c0>"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from IPython.display import YouTubeVideo\n",
    "YouTubeVideo('9IyZikbVlr4')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "deletable": true,
    "editable": true
   },
   "source": [
    "### Files Included"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "deletable": true,
    "editable": true
   },
   "source": [
    "Below are the list of files included in the project submission:\n",
    "    1. train.ipynb - Main Model (containing the script to create and train the model)\n",
    "    2. model_Final_Nvida_Bri.h5  -- containing a trained convolution neural network from train.ipynb\n",
    "    3. drive.py - For driving the car in autonomous mode\n",
    "    4. video.py - Used to create the simulation testing into vidoe file\n",
    "    5. Final_Nvida_Bri.mp4 -- Vidoe of the simulation test run\n",
    "    6. Behavioral Cloning-Report.ipynb -- Project Report (writeup_report summarizing the results)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true,
    "deletable": true,
    "editable": true
   },
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.5.2"
  },
  "widgets": {
   "state": {},
   "version": "1.1.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
