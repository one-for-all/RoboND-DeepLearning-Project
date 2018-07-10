## Project: Follow Me - Deep Learning Perception

---


# Required Steps for a Passing Submission:
1. Clone the project repo [here](https://github.com/udacity/RoboND-DeepLearning-Project.git).
2. Fill out the project code for training the model.
3. Optimize your network and hyper-parameters.
4. Train your network and achieve an accuracy of 40% (0.40) using the Intersection over Union IoU metric which is `final_grade_score` at the bottom of your notebook.
5. Make a brief writeup report summarizing why you made the choices you did in building the network.

[//]: # (Image References)
[encoder block]: misc_images/encoder_block.png
[decoder block]: misc_images/decoder_block.png
[network]: misc_images/network.png
[model]: misc_images/model.png
[final grade score]: misc_images/final_grade_score.png
[loss graph]: misc_images/loss_graph.png

[following]: misc_images/following.png
[patrol no target]: misc_images/patrol_no_target.png
[patrol target]: misc_images/patrol_target.png

[//]: # (Hyperlink References)

## [Rubric](https://review.udacity.com/#!/rubrics/1155/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup

#### 1. Provide a write-up / README document including all rubric items addressed in a clear and concise manner.

This document!

#### 2. The write-up conveys an understanding of the network architecture.

Firstly, some building blocks are defined here:

**Encoder Block**

![Encoder Block Diagram][encoder block]

 * The separable convolution has a `stride` of 2, `kernel size` of 3, `relu` activation, to reduce the dimension of next layer by half, and extract features.
 * Batch normalization is used to normalize input to each layer, to help with training, and help tackle overfitting.
 * The dropout has a `drop rate` of 0.5 to tackle overfitting.

**Decoder Block**

![Decoder Block Diagram][decoder block]

* Bilinear upsampling is used to upsample the features by 2, while restraining the number of new parameters to learn.
* Next, the upsampled layer are concatenated with a previous layer to gather fine-grained information from previous features.
* Another separable convolution with `stride` of 1, `kernel size` of 3, `relu` activation, is applied to extract new features from the concatenated layers.
* Batch normalization is again applied to normalize input to next layer.
* Dropout with `drop rate` of 0.5 is applied to tackle overfitting.

Using these building blocks, the entire network is built as follows:

![Neural Network Diagram][network]

* The 3 encoder block has `depth` of 32, 64 and 128 respectively, to extract features successively.
* The 1x1 convolution of `depth` 128 with batch normalization and dropout is used to extract features from encoder, while preserving spatial information.
* The first decoder block has no concatenation, and has `depth` of 128.
* The second decoder block has concatenation with first encoder block, and has `depth` of 64.
* The third decoder block has concatenation with input, and has `depth` of 32.
* Lastly, the decoder is convoluted with `kernel size` 3 and activated by `softmax`, to give a layer of `depth` 3, which is the number of categories (background, other people, target people).

An image of the model generated using keras is attached below for completeness:

![Model Image][model]

To allow the network to generalize to situations, additional data were collected using the simulator for various scenarios. In the end, ~24,000 training images were used.

#### 3. The write-up conveys the student's understanding of the parameters chosen for the the neural network.

The hyper-parameters are fine-tuned by training the network and observing the evolution of training loss and validation loss.

* Epoch = 11, because the validation loss has stopped decreasing significantly at around this number.
* Learning Rate = 0.001, the training loss is able to drop fast enough with this learning rate, and learning rate higher than this would make the converged training loss higher, while learning rate lower than this does not provide significant reduction in training loss.
* Batch Size = 16, because although low batch size would decrease training speed, we found that small batch size was able to help tackle the overfitting problem.
* Steps Per Epoch = total number of training images / batch size, so that about every image has a chance to be trained in each epoch.
* validation_steps = total number of validation images / batch size, so that the entire validation set is utilized to give more accurate validation score.
* Number of workers used in training = 4, because the AWS machine has 4 cores.

A loss graph of the training process is attached here:

![Loss Graph][loss graph]

#### 4. The student has a clear understanding and is able to identify the use of various techniques and concepts in network layers indicated by the write-up.

A comparison of 1x1 convolution and fully connected layer is discussed here:

**1x1 convolution:**
 * 1x1 convolution is just a regular convolution where kenel size = 1, and stride = 1. 
 * This is used to extract features for each pixel from all of its channels, and condense them to a smaller number of channels/depth.
 * It helps with reducing the number of features used for next layer, while preserving spatial information.
 * In the above network, it is used to condense features from encoder before passing to decoder.

**fully connected layer:**
* A Fully connected layer is often used to generate a classification for the entire input.
* With CNN, it is usually applied after the previous convolution layer has been flattened. Therefore, spatial information would be lost.
* In the above network, it is not used because we need to generate classification for each pixel, and spatial information is essential for that.

#### 5. The student has a clear understanding of image manipulation in the context of the project indicated by the write-up.

The task of this project is semantic segmentation, i.e. labeling each pixel of image by the category it belongs.

The network is constructed as a encoder - deocoder structure with skip connections between them.

Encoder is used to extract features relevant of the task, such as lines, shapes, objects, successively.

Decoder then takes these features and reconstruct the proper classification for each pixels.

Skip connections between encoder and decoder are used, because as the features are extracted in encoder, spatial information is condensed and fine details in the image are lost. Skip connections would provide previous layers' information where details are preserved to the later construction of categories for pixels.

#### 6. The student displays a solid understanding of the limitations to the neural network with the given data chosen for various follow-me scenarios which are conveyed in the write-up.

To change the task to follow another object (dog, cat car, etc.), the same setup and model could be used, because in the project, no domain knowledge of `person` was used.

However, to fulfill the task, we need to collect data on the target object, where this target is masked with its category. And the model needs to be retrained with the new data.

### Model

#### 1. The model is submitted in the correct format.

The model and its weights are saved in `data/weights/` as `config_follow_me_model.h5` and `follow_me_model.h5`.

#### 2. The neural network must achieve a minimum level of accuracy for the network implemented.

The final score is:

![Final Grade Score image][final grade score]

### Future Enhancements

* One important limitation of the project is that it is coded in a way that is specific for a single task. In the future, the project could be extended where new target objects could be added easily.

* As can be seen by the score, and in the example segmentations below, the trained model is still far from perfect. To improve it, more amount and more diverse data could be collected, and then more complex neural network model could be constructed.

* Another enhancement is to apply the model to real world images, such that the segmentation could be used by actual cameras, and possibly by real drones. An obstacle to this is obtaining ground truth segmentations.

### Demo of Semantic Segmentation

Three situations for semantic segmentation are shown here.    
* left image: image seen by the drone.    
* middle image: ground truth of segmentation.
* right image: predicted segmentation.

During following the target:

![Follow Target image][following]

Patrolling and target is not seen:

![Patrol No Target image][patrol no target]

Patrolling and target is seen:

![Patrol Target image][patrol target]

### Demo video of Follow Me

A demo video of executing the task of following is provided at this [youtube link](https://youtu.be/ktDXhRLbAQY).