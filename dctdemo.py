#!/usr/bin/env python

"""
wxpython GUI around a DCT demo
"""

import wx
import re
import numpy as np
from scipy.fftpack import dct, idct
import scipy.misc

# The recommended way to use wx with mpl is with the WXAgg
# backend.
import matplotlib
matplotlib.use('WXAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_wxagg import \
    FigureCanvasWxAgg as FigCanvas, \
    NavigationToolbar2WxAgg as NavigationToolbar

import matplotlib.pyplot as plt

class DctDemo(wx.Frame):
    """
    The main frame of the application
    """
    title = 'DCT Demo'

    def __init__(self):
        wx.Frame.__init__(self, None, -1, self.title)

        self.unzig = np.array([
            0,  1,  8, 16,  9,  2,  3, 10,
            17, 24, 32, 25, 18, 11,  4, 5,
            12, 19, 26, 33, 40, 48, 41, 34,
            27, 20, 13,  6,  7, 14, 21, 28,
            35, 42, 49, 56, 57, 50, 43, 36,
            29, 22, 15, 23, 30, 37, 44, 51,
            58, 59, 52, 45, 38, 31, 39, 46,
            53, 60, 61, 54, 47, 55, 62, 63])

        self.lena = scipy.misc.lena()
        self.create_status_bar()
        self.create_main_panel()

    def create_main_panel(self):
        """
        Creates the main panel with all the controls on it:
            * mpl canvas
            * mpl navigation toolbar
            * Control panel for interaction
        """
        self.panel = wx.Panel(self)

        self.sld = wx.Slider(self.panel, -1, 1, 1, 64, wx.DefaultPosition, (250, -1),
                wx.SL_AUTOTICKS | wx.SL_HORIZONTAL | wx.SL_LABELS)

        self.Bind(wx.EVT_SCROLL_CHANGED, self.on_adjust, self.sld)

        # Create the mpl Figure and FigCanvas objects.
        self.fig = Figure((5.0, 4.0))
        self.fig, self.axes = plt.subplots(nrows=2, ncols=2)
        self.canvas = FigCanvas(self.panel, -1, self.fig)

        # Create the navigation toolbar, tied to the canvas
        self.toolbar = NavigationToolbar(self.canvas)

        # Layout with box sizers
        self.vbox = wx.BoxSizer(wx.VERTICAL)
        self.vbox.Add(self.toolbar, 0, wx.EXPAND)
        self.vbox.Add(self.canvas, 1, wx.LEFT | wx.TOP | wx.GROW)
        self.vbox.AddSpacer(10)

        flags = wx.ALIGN_LEFT | wx.ALL | wx.ALIGN_CENTER_VERTICAL

        self.hbox = wx.BoxSizer(wx.HORIZONTAL)
        self.hbox.Add(self.sld, 0, border=3, flag=flags)
        self.vbox.Add(self.hbox, 0, flag = wx.ALIGN_CENTER | wx.TOP)

        self.panel.SetSizer(self.vbox)
        self.vbox.Fit(self)

        self.on_adjust()

    def create_status_bar(self):
        """
        A statusbar is nice to have
        """
        self.statusbar = self.CreateStatusBar()
        self.statusbar.SetStatusText('DCT Demo')

    def hinton(self, max_weight=None, ax=None):
        """
        Draw Hinton diagram for visualizing a weight matrix
        """
        quant = self.sld.GetValue()
        matrix = np.zeros(64) - 0.5
        matrix[self.unzig[:quant]] = 0.5
        matrix = matrix.reshape([8,8])

        if not max_weight:
            max_weight = 2**np.ceil(np.log(np.abs(matrix).max())/np.log(2))

        ax.patch.set_facecolor('gray')
        ax.set_aspect('equal', 'box')
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())

        for (x,y),w in np.ndenumerate(matrix):
            color = 'green' if w > 0 else 'black'
            size = np.sqrt(np.abs(w))
            # origo is at lower left instead of top left, transform coordinates
            rect = plt.Rectangle([x - size / 2, 7-y - size / 2], size, size,
                                 facecolor=color, edgecolor=color)

            ax.add_patch(rect)

        ax.autoscale_view()

    def psnr(self, a, b):
        m = ((a - b) ** 2).mean()
        if m == 0:
            return float("nan")

        return 10 * np.log10(255 ** 2 / m)

    def compute_ssim(self, img_mat_1, img_mat_2):
        """
        implementation using scipy and numpy from
        http://isit.u-clermont1.fr/~anvacava/code.html
        by antoine.vacavant@udamail.fr
        Usage by kind permission from author.
        """
        import scipy.ndimage
        from numpy.ma.core import exp
        from scipy.constants.constants import pi

        #Variables for Gaussian kernel definition
        gaussian_kernel_sigma = 1.5
        gaussian_kernel_width = 11
        gaussian_kernel = np.zeros((gaussian_kernel_width, gaussian_kernel_width))

        #Fill Gaussian kernel
        for i in range(gaussian_kernel_width):
            for j in range(gaussian_kernel_width):
                gaussian_kernel[i, j] = \
                    (1 / (2 * pi * (gaussian_kernel_sigma ** 2))) *\
                    exp(-(((i-5)**2)+((j-5)**2))/(2*(gaussian_kernel_sigma**2)))

        #Convert image matrices to double precision (like in the Matlab version)
        img_mat_1 = img_mat_1.astype(np.float)
        img_mat_2 = img_mat_2.astype(np.float)

        #Squares of input matrices
        img_mat_1_sq = img_mat_1 ** 2
        img_mat_2_sq = img_mat_2 ** 2
        img_mat_12 = img_mat_1 * img_mat_2

        #Means obtained by Gaussian filtering of inputs
        img_mat_mu_1 = scipy.ndimage.filters.convolve(img_mat_1, gaussian_kernel)
        img_mat_mu_2 = scipy.ndimage.filters.convolve(img_mat_2, gaussian_kernel)

        #Squares of means
        img_mat_mu_1_sq = img_mat_mu_1 ** 2
        img_mat_mu_2_sq = img_mat_mu_2 ** 2
        img_mat_mu_12 = img_mat_mu_1 * img_mat_mu_2

        #Variances obtained by Gaussian filtering of inputs' squares
        img_mat_sigma_1_sq = scipy.ndimage.filters.convolve(img_mat_1_sq, gaussian_kernel)
        img_mat_sigma_2_sq = scipy.ndimage.filters.convolve(img_mat_2_sq, gaussian_kernel)

        #Covariance
        img_mat_sigma_12 = scipy.ndimage.filters.convolve(img_mat_12, gaussian_kernel)

        #Centered squares of variances
        img_mat_sigma_1_sq = img_mat_sigma_1_sq - img_mat_mu_1_sq
        img_mat_sigma_2_sq = img_mat_sigma_2_sq - img_mat_mu_2_sq
        img_mat_sigma_12 = img_mat_sigma_12 - img_mat_mu_12

        #c1/c2 constants
        #First use: manual fitting
        c_1 = 6.5025
        c_2 = 58.5225

        #Second use: change k1,k2 & c1,c2 depend on L (width of color map)
        l = 255
        k_1 = 0.01
        c_1 = (k_1 * l) ** 2
        k_2 = 0.03
        c_2 = (k_2 * l) ** 2

        #Numerator of SSIM
        num_ssim = (2 * img_mat_mu_12 + c_1) * (2 * img_mat_sigma_12 + c_2)
        #Denominator of SSIM
        den_ssim = (img_mat_mu_1_sq + img_mat_mu_2_sq + c_1) *\
            (img_mat_sigma_1_sq + img_mat_sigma_2_sq + c_2)
        #SSIM
        ssim_map = num_ssim / den_ssim
        index = np.average(ssim_map)

        return index

    def on_adjust(self, event=None):
        """
        Executed when slider changes value
        """
        width, height = self.lena.shape
        quant = self.sld.GetValue()

        # reconstructed
        rec = np.zeros(self.lena.shape, dtype=np.int64)

        for y in xrange(0,height,8):
            for x in xrange(0,width,8):
                d = self.lena[y:y+8,x:x+8].astype(np.float)
                D = dct(dct(d.T, norm='ortho').T, norm='ortho').reshape(64)
                Q = np.zeros(64, dtype=np.float)
                Q[self.unzig[:quant]] = D[self.unzig[:quant]]
                Q = Q.reshape([8,8])
                q = np.round(idct(idct(Q.T, norm='ortho').T, norm='ortho'))
                rec[y:y+8,x:x+8] = q.astype(np.int64)

        diff = np.abs(self.lena-rec)

        im = self.axes.flat[0].imshow(self.lena, cmap='gray')
        self.axes.flat[0].set_title('Original Image')

        im = self.axes.flat[1].imshow(rec, cmap='gray')
        self.axes.flat[1].set_title('Reconstructed Image')

        self.hinton(ax=self.axes.flat[2])
        self.axes.flat[2].set_title('DCT Coefficient Mask')

        im = self.axes.flat[3].imshow(diff, cmap='hot', vmin=0, vmax=255)
        self.axes.flat[3].set_title('Error Image')

        for ax in self.axes.flat:
            ax.axis('off')

        self.fig.subplots_adjust(right=0.8)
        cbar_ax = self.fig.add_axes([0.85, 0.15, 0.05, 0.7])
        self.fig.colorbar(im, cax=cbar_ax)

        p = self.psnr(self.lena, rec)
        s = self.compute_ssim(self.lena, rec)

        self.statusbar.SetStatusText('PSNR :%.4f SSIM: %.4f' % (p, s))

        self.canvas.draw()

    def on_exit(self, event):
        self.Destroy()

if __name__ == '__main__':
    app = wx.PySimpleApp()
    app.frame = DctDemo()
    app.frame.Show()
    app.MainLoop()
