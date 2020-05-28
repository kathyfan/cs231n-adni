# Just need torch.tensor packages to pass in tensors
import torch
from torch.Tensor 

# define a dummy layer
# Call as: feature_dense_enc_masked = BinaryMask(32,mask,pre_feature)(feature_dense_enc)
# feature_dens_enc should be the flatted conv layer, 32 is an output dimension size
# Next call in notebook is classifier(feature_dense_enc_masked)
# Classifier is a model, so your variable for the binary mask should be passed into a Model() variable
class BinaryMask(Layer):

    def __init__(self, output_dim, mask, pre_feature, **kwargs):
        self.output_dim = output_dim
        self.mask = mask
        self.pre_feature = pre_feature
        
        super(BinaryMask, self).__init__(**kwargs)

    def build(self, input_shape):
        self.mask_tensor = torch.tensor(self.mask)
        self.pre_feature = torch.tensor(self.pre_feature)
        
        super(BinaryMask, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return x * self.mask_tensor + self.pre_feature

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)