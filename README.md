## A Decoupled, Generative, Unsupervised, Multimodal Architecture For Real-World Agents.

![Architecture](https://github.com/jakobovski/decoupled-multimodal-learning/raw/master/assets/reward_architecture_schematic.png "Architecture")


![Audio Encodings](https://github.com/jakobovski/decoupled-multimodal-learning/raw/master/assets/tsne_audio_small.png "TSNE Audio encodings")
![MNIST Encodings](https://github.com/jakobovski/decoupled-multimodal-learning/raw/master/assets/tsne_visual_small.png "TSNE MNIST encodings")



## What is this repository? 
This repository contains a working implementation of a new generative multimodal architecture. Totally unsupervised, and with large invariance to hyperparameters, this implementation learns to classify MNIST in terms of [FSDD](https://github.com/Jakobovski/free-spoken-digit-dataset)(audio version of MNIST) and vice versa with `~97.8%` accuracy and discovers the correct number of classes within the data. 
 

## TLDR: Why this architecture is neat.
1. It learns to classify data accurately without supervision and without prior knowledge of the classes contained in the data. It does this by leveraging temporal correlations between multimodal "stimuli". Similar to a baby learning the names of objects by correlating the image of the object with the sound coming from its parent. 
2. It discovers the number of classes in the data much more robustly than classical clustering algorithms.
3. Learned knowledge is grounded. Data is learned and classified in terms of other data, IE: images in-terms of sounds, images in terms of other images, etc..
4. It is highly extensible. Any number of additional modalities can easily be added. 
5. 'Rewards' can be treated as a modality and the system will learn to associate classes with their expected reward.
6. It's simple and biologically plausible. See: [Trends in Neurosciences: convergence-divergence-zone](http://www.cell.com/trends/neurosciences/fulltext/S0166-2236(09)00090-3).


### Convergence-Divergence Zone
The convergence-divergence zone, proposed by Antonio Damasio in 1989 is a biologically plausible and [experimentally supported](http://www.cell.com/trends/neurosciences/fulltext/S0166-2236(09)00090-3) architecture that describes how the brain learns generative, multimodal representations.  Due to this architecture's similarities with the Damasio's CDZ, I use the term CDZ throughout the readme. Damasio described the CDZ on a high level, the CDZ presented in this repository *mostly* conforms to his high level description with the algorithmic details filled in as I saw fit.


______________________________________________

## Summary
Autonomous agents need the ability to acquire grounded knowledge from multiple sensory channels in an unsupervised way. 
Most approaches learn high dimensional representations that are shared between modalities. These approaches suffer from 
a number of drawbacks including: cross-modality variance, (variance in one modality cause generative variances in the
other modalities), computational inefficiencies, strict temporal requirements, and inability to model one-to-many 
relationships. In this project I tackle the problem of unsupervised learning from an agent's environment. I present an
architecture that solves these issues by decoupling the modalities thus providing a more robust and extensible 
foundation for modeling multimodal data. I demonstrate the architecture by learning to classify MNIST in terms of the 
FSDD audio dataset (and vice versa). Using only unsupervised raw input, and with large invariance to hyperparameters, 
the model learns the optimal one-of-k representation of the input, discovering `k=10`, the exact number of 
classes in the raw data. The model achieves a competitive classification accuracy as measured by the one-of-k clusters, 
removing the need for a supervised layer in measuring classification accuracy. 



## Introduction / How it works.
Multimodal learning involves relating information from multiple sensory sources. This type of learning is particularly useful for real-world applications since information in the real world is multimodal. In the real world there exist significant semantic correlations between the modalities. For example a visual image of a dog often coincides with the sound of a bark, the pronunciation of the english word "dog", the texture of fur etc.. If a generative agent can learn to associate the image of a dog with the English pronunciation of "dog" then it has learned to classify. In other words, these multimodal correlations can be used as soft labels and provide a rich learning signal.

In addition to providing a learning signal, multimodal information, specifically generative representations, provide a means of grounding as knowledge in one modality is represented in terms of one or more other modalities.  When such an agent hears the word “dog”, it is able to generate a visualization, a reconstruction of the feeling of its texture, and possibly generation of its expected reward from the dog (if one encapsulates reward information into a modality). Such representations give an agent knowledge of its environment that is far richer and more executable than one-of-k representations. 


## Architecture
The architecture consists of two or more modalities and one convergence-divergence zone (CDZ). The modalities each consist of a deep, neural autoencoder and a clustering component. For the rest of this readme, unless stated otherwise I describe an architecture consisting of only two modalities. 


![Architecture](https://github.com/jakobovski/decoupled-multimodal-learning/raw/master/assets/architecture_schematic.png "Architecture")

*An example implementation. Raw sensory data is passed to an autoencoder and is converted to an n-dimensional encoding. The nearest node is excited, which in turn excites its most correlated cluster. The cluster excites the CDZ which starts the process in reverse in the other modality. Each cluster represents a class. The CDZ learns to associate the clusters from different modalities based on their temporal proximity.*


## Autoencoders
There are many types of autoencoders, all of which are compatible with this architecture. I used stacked, denoising autoencoders, which are not state-of-the-art. These produced low-quality encodings which allowed me to properly test the architecture. (It is not very challenging to classify encodings if the are of a very high quality)


![Audio Encodings](https://github.com/jakobovski/decoupled-multimodal-learning/raw/master/assets/tsne_audio_small.png "TSNE Audio encodings")
![MNIST Encodings](https://github.com/jakobovski/decoupled-multimodal-learning/raw/master/assets/tsne_visual_small.png "TSNE MNIST encodings")

*TSNE visualizations of the 64-dimensional encodings of FSDD(left) and MNIST(right) training sets that were used in the experiments. Labels are distinguished by color.*



## Clustering Algorithm
A clustering algorithm is used to find regions of encoding space that correspond to labels. Standard unsupervised clustering algorithms such as K-means or Growing Neural Gas are at the mercy of the precision of the autoencoders and the statistical properties of the data itself. For example in TSNE visualizations above (while ignoring color) the boundaries between many of the digits are not well defined and are unlikely to be discovered accurately using traditional clustering techniques.

Given multimodal information, it makes sense to leverage it for learning clusters. I therefore introduce a new clustering algorithm that is inherently multimodal. Like all clustering algorithms there are no guarantees that it will find the correct number of clusters(assuming such a number even exists). Despite a lack of guarantees I found experimentally that the algorithm converges to finding the correct number of clusters over a surprisingly wide range of hyperparameters and data distributions.

<img src="https://github.com/jakobovski/decoupled-multimodal-learning/raw/master/assets/clustering_algo.png" width="400">

<img src="https://github.com/jakobovski/decoupled-multimodal-learning/raw/master/assets/maintenance_algo.png" width="400">

The clustering algorithm is dependent on the assumption that regions in a modality's encoding space that represent the same class, have similar co-occurrence distributions with the clusters of other modalities. For example all the points in the visual encoding space that represent “dog”, have approximately the same probability of being excited at the same time as the audio cluster of “dog”. If a point in the visual space has some probability of being excited with audio “dog” and audio “cat” then it indicates that the point lies in between the visual “dog” and “cat” manifolds. In such a case, new nodes would be created in that borderline region of space until the boundaries between classes are well defined and accurate.

The algorithm has a very nice property such that when used for generative purposes, the encodings generated are not the mean representations or the center point of the cluster, but rather an approximation of the mode. This is ideal because for real-world data, the mode is less noisy than the mean, and is a more useful for generative purposes. 

The algorithm can be interpreted as using Hebbian learning to associate encodings, represented by their nearest nodes, with clusters. This is because the excitation of a cluster in one modality triggers the excitation of the clusters in the other modalities that it is connected to.


## Convergence-Divergence Zone
The CDZ provides a number of valuable services:
- A sparse, semantic interface between modalities.
- A means of decoupling modalities.
- A means of correlating, one-to-many stimuli between modalities.

The CDZ used in this experiment is described by the algorithm below:

<img src="https://github.com/jakobovski/decoupled-multimodal-learning/raw/master/assets/cdz_algo.png" width="400">


The CDZ learns to correlate clusters by their temporal proximity. The more often two clusters are excited in temporal proximity the stronger their mutual connections become. To be precise the more often that the excitation of cluster C2, precedes the excitation of cluster C1, (within the timeframe specified by `B`) the stronger the connections from C1 to C2 become. Note that the connection from C2 to C1 is unaffected (except when t=0), in other words connections-strengths are not symmetrical. This one-directional rule is somewhat arbitrary and is not a requirement for all CDZ implementations. I chose it because it appears to be how the mammalian brain works. For example: Pavlov's dogs.

Although not described in the CDZ algorithm above, it should also be noted that the CDZ can be modified such that clusters can excite other clusters within the same modality. I think this may be useful for learning invariance, for as the sensory data(encodings) changes with translation, rotation etc.. nodes in the new encoding regions can leverage the previously excited same modality cluster to learn their appropriate cluster, and thus learn invariance.

## Experiments
In order to test the architecture I built an implementation with an audio and a visual modality  and trained it using MNIST and FSDD datasets.


#### Datasets
[FSDDv1.0](https://github.com/Jakobovski/free-spoken-digit-dataset/releases/tag/v1.0) consists of 500, 8 kHz recordings of English pronunciations of the digits 0-9. These recordings are of varying durations, with a mean of approximately 0.5s. I preprocessed the recordings by converting them into 64x64 grayscale spectrograms using FSDD's bundled toolkit. I split the dataset into a training and test set of 450 and 50 recordings respectively. 

MNIST is a dataset consisting of images of handwritten digits. The set consists of 55,000 training and 10,000 test grayscale images, with dimensions of 28x28 pixels. I did not perform any preprocessing on this dataset.

####  Autoencoder setup
For dimensionality reduction I used deep denoising autoencoders consisting of [2048, 1024, 256, 64] neurons per layer for MNIST, and [4096, 512, 64] neurons per layer for FSDD. Both autoencoders output a 64 dimensional encoding. ADAM optimizer was used with an initial learning rate or 1e-4. Hyperparameters were identical in both modalities.

The autoencoders were first trained in isolation, and then the were connected to the clustering and CDZ components. Initial isolation is not a requirement, but it is a good practice in order to decrease total training time, as there is no point wasting computation by training the clusters and CDZ on encodings that are far from their converged values. 

#### CDZ Setup
The CDZ algorithm used was similar to that described above. For this experiment I set `t=0`, so that only co-occurring stimuli were learned. See code for full details.

## Results
The autoencoders produced relatively low quality MNIST encodings, this was done intentionally to demonstrate the efficacy of the algorithm, as high classification on clean data is not very challenging even for unsupervised algorithms. The architecture discovered exactly 10 unique classes in both the visual and audio modalities, one for each digit. The table below compares the presented architecture to a 2-layer neural network on the classification accuracy of encodings.

| Measured On          | Architecture                     | Accuracy  |
| -------------------- | -------------------------------- | --------- |
| MNIST test encodings | Autoencoder + CDZ (unsupervised) | 95.6%     |
| FSDD test encodings  | Autoencoder + CDZ (unsupervised) | 100%      |
| MNIST test encodings | 2-layer classifier, 32 hidden units (supervised) | 96.4%     |
| FSDD test encodings  | 2-layer classifier, 32 hidden units (supervised) | 100%      |


Accuracy was measured by determining the frequency in which a stimuli of a given label triggered the generation of identical data in the second modality.


## Future work / Things to explore
The code in this repository is proof-of-concept and is rather limited in and of itself. 

I am interested in exploring functional extensions of the architecture in the direction of extending it into a complete cognitive architecture. Future work will be directed towards (1) using second modality information to train the autoencoders, (2) modifying the clustering algorithm such that it clusters manifolds using a Hebbian neural networks, (3) using RNN auto-encoders, (4) learning classifications from watching children's videos.


## Last words
One final neat idea: If the generative output of each modality is looped back into its encoder then the architecture will exhibit something similar to an endless 'stream-of-thought'. Excitation in one modality will lead to excitation in the same or a different modality and because the system is generative, it will actually "see" and "hear" its own excitations. Although this is not very practical, its similarities to human abilities to re-create sensory phenomenon through imagination and memory is neat.



___________________________________________________

## Code and Installation


##### How to use this repository:
Before using the repository make sure to install all the requirements.
```bash 
$ pip install -r requirements.pip
```


##### Generating encodings
```bash 
$ python utils/fsdd_encoding_generator.py
$ python utils/mnist_encoding_generator.py
```


##### Running the complete algorithm
```bash 
$ python examples/basic_example.py
```


##### Generating TSNE visualization of the encodings
```bash 
$ python utils/tsne_generator.py
```
