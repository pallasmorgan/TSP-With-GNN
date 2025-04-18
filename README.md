Assignment 4

edits made by aylene:
[10] pre-established target value outside the loop, updated target to have a parameter device = device, added a checker to make sure tensor and models are on the same device before entering the loop
[11] changed visited as a torch.BoolTensor, set the device to next(model.parameters()).device, masked_fill(visited, -inf), removed the current depedency
[12] set lats,lons =zip(*coords), offset text(+0.002), tight_layout(), and added linestyle='-' to the plt.plot parameters
