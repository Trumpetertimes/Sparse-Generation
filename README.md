# Sparse Generation: Making Pseudo Labels Sparse for point weakly supervised object detection on low data volume

![image](https://github.com/Trumpetertimes/Sparse_Generation/blob/master/SP_pipeline0912.png)
Sparse Generation uses non-networked approach and direct regression on pseudo labels. In three processing stages (Mapping, Mask, Regression), Sparse Generation constructs initial tensors through the relationship between data and detector model, optimizes its parameters, and obtains a sparse tensor, addresses the model’s density problem on low data volume. 
