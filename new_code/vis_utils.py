from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation
import os
import json

import nibabel as nib
from scipy.ndimage.interpolation import zoom
from scipy import ndimage


# Transparent colormap (alpha to red), that is used for plotting an overlay.
# See https://stackoverflow.com/questions/37327308/add-alpha-to-an-existing-matplotlib-colormap
alpha_to_red_cmap = np.zeros((256, 4))
alpha_to_red_cmap[:, 0] = 0.8
alpha_to_red_cmap[:, -1] = np.linspace(0, 1, 256)#cmap.N-20)  # alpha values
alpha_to_red_cmap = mpl.colors.ListedColormap(alpha_to_red_cmap)

red_to_alpha_cmap = np.zeros((256, 4))
red_to_alpha_cmap[:, 0] = 0.8
red_to_alpha_cmap[:, -1] = np.linspace(1, 0, 256)#cmap.N-20)  # alpha values
red_to_alpha_cmap = mpl.colors.ListedColormap(red_to_alpha_cmap)

def plot_slices(struct_arr, num_slices=7, cmap='gray', vmin=None, vmax=None, overlay=None, overlay_cmap=alpha_to_red_cmap, overlay_vmin=None, overlay_vmax=None):
    """
    Plot equally spaced slices of a 3D image (and an overlay) along every axis
    
    Args:
        struct_arr (3D array or tensor): The 3D array to plot (usually from a nifti file).
        num_slices (int): The number of slices to plot for each dimension.
        cmap: The colormap for the image (default: `'gray'`).
        vmin (float): Same as in matplotlib.imshow. If `None`, take the global minimum of `struct_arr`.
        vmax (float): Same as in matplotlib.imshow. If `None`, take the global maximum of `struct_arr`.
            # When using scalar data and no explicit norm, vmin and vmax define the data range that the colormap covers. 
            # By default, the colormap covers the complete value range of the supplied data. 
            # vmin, vmax are ignored if the norm parameter is used.
        overlay (3D array or tensor): The 3D array to plot as an overlay on top of the image. Same size as `struct_arr`.
        overlay_cmap: The colormap for the overlay (default: `alpha_to_red_cmap`). 
        overlay_vmin (float): Same as in matplotlib.imshow. If `None`, take the global minimum of `overlay`.
        overlay_vmax (float): Same as in matplotlib.imshow. If `None`, take the global maximum of `overlay`.
    """
    if vmin is None:
        vmin = struct_arr.min()
    if vmax is None:
        vmax = struct_arr.max()
    if overlay_vmin is None and overlay is not None:
        overlay_vmin = overlay.min()
    if overlay_vmax is None and overlay is not None:
        overlay_vmax = overlay.max()
    print(vmin, vmax, overlay_vmin, overlay_vmax)

    fig, axes = plt.subplots(3, num_slices, figsize=(15, 6))
    intervals = np.asarray(struct_arr.shape) / num_slices

    for axis, axis_label in zip([0, 1, 2], ['x', 'y', 'z']):
        for i, ax in enumerate(axes[axis]):
            i_slice = int(np.round(intervals[axis] / 2 + i * intervals[axis]))
            # print(axis_label, 'plotting slice', i_slice)
            
            plt.sca(ax)
            plt.axis('off')
            plt.imshow(sp.ndimage.rotate(np.take(struct_arr, i_slice, axis=axis), 90), vmin=vmin, vmax=vmax, 
                       cmap=cmap, interpolation=None)
            plt.text(0.03, 0.97, '{}={}'.format(axis_label, i_slice), color='white', 
                     horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
            
            if overlay is not None:
                plt.imshow(sp.ndimage.rotate(np.take(overlay, i_slice, axis=axis), 90), cmap=overlay_cmap, 
                           vmin=overlay_vmin, vmax=overlay_vmax, interpolation=None)



def resize_image(img, size, interpolation=0):
    """Resize img to size. Interpolation between 0 (no interpolation) and 5 (maximum interpolation)."""
    zoom_factors = np.asarray(size) / np.asarray(img.shape)
    return sp.ndimage.zoom(img, zoom_factors, order=interpolation)