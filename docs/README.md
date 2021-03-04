---
layout: home
permalink: index.html

# Please update this with your repository name and title
repository-name: e15-4yp-Brain-Computer-Interface-for-controlling-virtual-objects
title: Brain Computer Interface for controlling virtual objects
---

[comment]: # "This is the standard layout for the project, but you can clean this and use your own template"

# Project Title

#### Team

- E/15/023, Avishka Athapattu, [email](mailto:e15023@eng.pdn.ac.lk)
- E/15/059, Prageeth Dassanayake, [email](mailto:e15059@eng.pdn.ac.lk)
- E/15/238, Sewwandie Nanayakkara, [email](mailto:sewwandiecn@gmail.com)

#### Supervisors

- Dr. Isuru Nawinne, [email](mailto:isurunawinne@eng.pdn.ac.lk)
- Prof. Roshan Ragel, [email](mailto:roshanr@eng.pdn.ac.lk)
- Theekshana Dissanayake, [email](mailto:theekshanadis@gmail.com)

#### Table of content

1. [Abstract](#abstract)
2. [Related works](#related-works)
3. [Methodology](#methodology)
4. [Experiment Setup and Implementation](#experiment-setup-and-implementation)
5. [Results and Analysis](#results-and-analysis)
6. [Conclusion](#conclusion)
7. [Publications](#publications)
8. [Links](#links)

---

## Abstract

<p>Non-invasive EEG based Brain Computer Interface (BCI) systems have been an interesting research area for many fields. However most of the research done on this subject is synchronous, therefore the state of mind of the user is not similar to its natural behaviour. Considering to provide possible experience in practical applications, self-paced BCI systems started gaining popularity in recent years. However, there are certain challenges yet to be addressed when following this method. Out of the research done on self-paced BCI systems most of them are focused on motor-imagery control whereas research on nonmotor imagery mental tasks is limited. In this research, we analyse the possibility of using the techniques used in the motorimagery method for non-motor imagery mental tasks to be fed into virtual object controlling applications.</p>

## Related works

<p>Both non-motor imagery EEG signals related to virtual object manipulation and motor imagery EEG signals are sensorimotor rhythms(SMR). These are specific brain waves over the sensorimotor cortex that are generated after MI or ME. In research by Faradji et.al [a link](https://cdn.intechopen.com/pdfs/65241.pdf), they explored the idea of rotation of a virtual object in 3D space in a more natural way. They used auto scalar auto-regressive methods for feature extraction and the classification was done with quadratic discriminant analysis. They obtained a true positive rate (TPR) value of 54.6% TPR and 0.01% FPR. Although there are numerous researches on using motor imagery to control virtual objects that give us higher accuracy [2], research done by Faradji et al. explores the possibility of controlling objects in a more natural way. It was stated that although the TPR is relatively low compared to MI related research, this method is more preferable in real-time applications since this method requires less computational power.
  
## Methodology

<p>First we trained the subject to train three mind intents which are left, right, and None without any visual aid. Afterwards,
we trained the subject with GUI aid. We used an OpenBCI Cyton board to capture EEG data in the experimental setup and signals were processed using Python. The number of EEG channels used was 8. EEG signals were fed for processing and denoising. We used the OpenBCI GUI to send EEG signals
through LSL (Lab Streaming Layer) into a Python application where we extracted the features. Our subject was a male volunteer, of age 24. Initially the subject performed a mental task while watching a virtual object on a screen. This training was done in a limited time trial like 0 -10 seconds, because the performance of the mental task degrades over time. </p>
  
<figure>
<img src="../code/Related_images/steps.jpg" width="500" height="400">
<figcaption>figure 1</figcaption>
</figure>
  
<h5>A. Electrodes and electrode placement</h5>
<p>We used eight Golden cup electrodes to sample EEG data. We placed those on the subject according to the 10-20 method. The 10–20 system or International 10–20 system is an internationally recognized method to describe and apply the location of scalp electrodes in the context of an EEG exam. EEGs were placed in 10% and 20% spaces on the scalp as follows. The brain waves related to controlling virtual objects are induced in the motor cortex so electrode placement positions are chosen so as to extract the maximum amount of information. In our experiment, we placed electrodes as shown in Fig. 2.</p>

<figure>
<img src="../code/Related_images/placements.jpg" width="500" height="400">
<figcaption>figure 2</figcaption>
</figure>

<h5>B.Virtual Environment</h5>
<p>Virtual objects that were meant for controlling are created with Unity. The subject is trained on a virtual environment where the display is 15.6 inch, monitor resolution of 1920 x 1080 p and 60Hz. Data of mind intent will be recorded where the subject will focus on moving the objects along axes. Shown in Fig. 3 is the virtual environment we created.</p>

<figure>
<img src="../code/Related_images/GUI_new.png" width="900" height="400">
<figcaption>figure 3</figcaption>
</figure>

## Experiment Setup and Implementation

## Results and Analysis
<p>Frequency bin components extracted by FFT and Detailed coefficients extracted by wavelet transform were used as features for the classification purpose. All the classifications have the ability to perform in real time. We used Random Forest, QDA, KNN, Catboost and SVM for classifying. In Table II we have compared the accuracies between different classification models. Table III gives the TPR of each class with respect to the model. The confusion matrix of the KNN model is shown in Fig. 8.


## Conclusion
<p>Filters that were used in EEG signal processing causes a phase shift that makes the usage of wavelet features impossible. Therefore we have used the FFT feature extraction method to provide frequency bins as features for our classification methods. But by substituting those filters (Butterworth filter) with others (zero phase filters) the effect of the phase shift can be removed. We can explore the possibility of using a combination of features provided by WT and FFT to train a more accurate classification model. With all the classification models that were trained KNN algorithm with FFT algorithm would be the ideal choice of features and classification combination. We were able to obtain around 55% TPR value. By implementing statistical analysis we can rectify the anatomical localization effects on EEG data would further increase accuracy of these models. Deep learning methods proved to have a lot of potential when it comes to MI based research in recent history. Possibility of using deep learning approaches in non motor imagery
intent with self phased brain computer interfaces is something that can be explored as well.</p>

## Publications
1. [Semester 7 report](./)
2. [Semester 7 slides](./)
3. [Semester 8 report](./)
4. [Semester 8 slides](./)
5. Author 1, Author 2 and Author 3 "Research paper title" (2021). [PDF](./).


## Links

[//]: # ( NOTE: EDIT THIS LINKS WITH YOUR REPO DETAILS )

- [Project Repository](https://github.com/cepdnaclk/e15-4yp-Brain-Computer-Interface-for-controlling-virtual-objects)
- [Project Page](https://cepdnaclk.github.io/e15-4yp-Brain-Computer-Interface-for-controlling-virtual-objects)
- [Department of Computer Engineering](http://www.ce.pdn.ac.lk/)
- [University of Peradeniya](https://eng.pdn.ac.lk/)

[//]: # "Please refer this to learn more about Markdown syntax"
[//]: # "https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet"
