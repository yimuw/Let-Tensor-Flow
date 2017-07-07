# Let Tensor Flow

Tensorflow is a great mathematical optimization framework. Most people only use it for machine learning. However, Tensorflow can do more than that! 

I wrote few applications of Tensorflow:
   1. Deep Learning (which is not very exciting)
   2. Graph Slam
   3. Optimal Control
   4. Calculus of Variations

The initial goal was to teach my wife using Tensorflow to do object detection. So, I tried to make the code straightforward. I hope you like it :)

## Getting Start

### Prerequisites

I am using python packages from Udacity self-driving car program. 

They are using anaconda environment to manage python packages. I recommend you to use it too because doing that is less likely to break your existing python packages.

How to set up python packages: https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md

If you want GPU support, you need to install Cuda libraries (which is that hardest part of deep learning) and run 
```
conda env create -f environment-gpu.yml
```  

After setup the environment, run this to activate the environment before running python scripts
```
source active carnd-term1
```
### Run 
#### deep learning
In this project, I try to detect object's 2d bird-eye location from images. I am using the Kitti detection dataset. I commit a sample dataset and a trained net to the repository (bad practice though). If you want to train the net, please download the full Kitti detection dataset and replace the sample dataset.

The visualize the trained net result, go to kitti_detection directory

```
python show_results.py
```

To train net, go to kitti_detection directory

```
python train_net.py
```

I didn't tune parameters or searching for network structure. So the result is not perfect (But I think it is good enough).

#### Graph Slam
I basically follow the graph slam chapter of probabilistic robotics. To simplify the problem and focus on the math side, I thesis data and assume association is known. But, data association is actually the hardest part of SLAM.

To run, go to slam directory
```
python graph_slam_with_know_association.py
```

#### Optimal Control
I design a non-linear investment system to illustrate the control problem. And how to solve it by Tensorflow.

To run, go to control directory
```
python optimal_control.py
```

#### Calculus of Variations

I solve Brachistochrone Curve by Tensorflow. 

"In mathematics and physics, a brachistochrone curve (from Ancient Greek βράχιστος χρόνος (brákhistos khrónos), meaning 'shortest time'),[1] or curve of fastest descent, is the one lying on plane between a point A and a lower point B, where B is not directly below A, on which a bead slides frictionlessly under the influence of a uniform gravitational field to a given end point in the shortest time. " 

Technically I didn't solve the calculus of variations using Tensorflow. I formulate a typical calculus of variations problem into an optimization problem. And I solve it byTensorflow.

To run, go to control directory
```
python brachistochrone_curve.py
```


## Authors

* **Yimu Wang** - *Initial work*


## License
TODO: what is MIT License?

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
