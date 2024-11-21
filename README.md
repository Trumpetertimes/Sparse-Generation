# Sparse Generation: Making Pseudo Labels Sparse for
point weakly supervised object detection on low data volume

Sparse Generation uses non-networked approach and direct regression on pseudo labels. In three processing stages (Mapping, Mask, Regression), Sparse Generation constructs initial tensors through the relationship between data and detector model, optimizes its parameters, and obtains a sparse tensor, addresses the model’s density problem on low data volume. 
