# Video-Frame-Prediction-using-PWC-Net


The objective behind this project is to implement a system which can predict future frames given the present and past frames.
The problem here is divided ito three main parts - 
1) Optical Flow computation using PWC-Net
2) Pyramidal decomposition of obtained flows for predicting future frames
3) Post processing for refinement of predicted frames

In our case, we start by taking first four frames to predict the target fifth frame. We design a pyramidal flow decomposition for these  four images and their flows. The frame closest to the target frame (in our case the fourth) frame is taken to be the anchor frame and flows are computed from every frame to this fourth frame. Once flows are obtained, we compute flows between any two arbitrary frames that are used for computation of velocity and acceleartion of a particular frame. This can be done by computing the first and second order derivatives. Once we obtain these, we do further calculations and find the flow between the fourth frame and fifth frame.
Once we have the anchor frame, and the flow between the anchor and target frame, we perform forward warping on on the flow and the anchor frame to obtain the target frame. The output here contains some occlusion holes as a result of forward warping. In order to remove these holes, we pass these frames through a trained post processing network which includes ResNet-50. The obtained result is the required target frame that is free of any occlusion holes.
  
