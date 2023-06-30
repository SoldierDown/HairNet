# Author: Shiyang Jia
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D

def show3Dhair(axis, strands):
    """
    strands: [100, 3, 32, 32]
    """
    for i in range(32*32):
        xs = []
        ys = []
        zs = []
        jj = int(i / 32)
        ii = i - 32 * jj
        if sum(sum(strands[:,:,ii,jj])):
            for sp in range(100):
                # transform from graphics coordinate to math coordinate
                y, z, x = strands[sp, 0, ii, jj], strands[sp, 1, ii, jj], strands[sp, 2, ii, jj] 
                # print('#{}, {}, {}: {}, {}, {}\n'.format(ii, jj, sp, x, y, z))
                xs.append(x)
                ys.append(y)
                zs.append(z)
                axis.plot(xs, ys, zs, linewidth=0.2, color='lightskyblue')
    RADIUS = 0.3  # space around the head
    xroot, yroot, zroot = 0.137028306722641, 0.3457982540130615, 1.753461480140686
    axis.set_xlim3d([-RADIUS+xroot, RADIUS+xroot])
    axis.set_ylim3d([-RADIUS+yroot, RADIUS+yroot])
    axis.set_zlim3d([-RADIUS+zroot, RADIUS+zroot])

    # Get rid of the ticks and tick labels
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_zticks([])

    axis.get_xaxis().set_ticklabels([])
    axis.get_yaxis().set_ticklabels([])
    axis.set_zticklabels([])
    axis.set_aspect('equal')

    """
    # Get rid of the panes (actually, make them white)
    white = (1.0, 1.0, 1.0, 0.0)
    axis.w_xaxis.set_pane_color(white)
    axis.w_yaxis.set_pane_color(white)
    axis.w_zaxis.set_pane_color(white)

    # Get rid of the lines in 3d
    axis.w_xaxis.line.set_color(white)
    axis.w_yaxis.line.set_color(white)
    axis.w_zaxis.line.set_color(white)
    """


def visualize_hair(name, image, strands_gt, strands_pred):
    # this produces 3x3 layout => wrong approach
    # np_image = image.numpy().squeeze() * 255
    # np_image = np.reshape(np_image, (256, 256, 3))
    # right way to go
    # cv_image = np.moveaxis(image.numpy().squeeze()*255, 0, -1)
    # cv2.imwrite('./output/cv2_' + name + '.png', cv_image)

    fig = plt.figure(figsize=(18, 6))
    fig.set_tight_layout(False)
    gs = gridspec.GridSpec(1, 3)
    gs.update(wspace=0.05)
    plt.axis('off')
    # plot orientation map
    ax1 = plt.subplot(gs[0])
    plt_image = np.moveaxis(image.numpy().squeeze(), 0, -1)
    # cv2 reads in BGR, not RGB
    plt_image = cv2.cvtColor(plt_image, cv2.COLOR_BGR2RGB)

    ax1.imshow(plt_image)
    
    # plot hair ground truth
    ax2 = plt.subplot(gs[1], projection='3d')
    # strands shape: [BATCH_SIZE, 100, 3, 32, 32]
    strands_gt = strands_gt.squeeze()
    show3Dhair(ax2, strands_gt)

    # plot predict hair
    ax3 = plt.subplot(gs[2], projection='3d')
    # strands shape: [BATCH_SIZE, 100, 3, 32, 32]
    strands_pred = strands_pred.squeeze()
    show3Dhair(ax3, strands_pred)
    plt.savefig('./output/plt_' + name + '.png')
