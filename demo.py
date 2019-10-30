# demo.py
#
# Fast Total-Variation-based JPEG Artifact Removal via the Accelerated ADMM
#
#
# Written by  : Chihiro Tsutake
# Affiliation : University of Fukui
# E-mail      : ctsutake@icloud.com
# Created     : October 2019
#

import scipy
import numpy
import skimage
import time

# Quantization table
Q = numpy.array([
    [16, 11, 10, 16,  24,  40,  51,  61],
    [12, 12, 14, 19,  26,  58,  60,  55],
    [14, 13, 16, 24,  40,  57,  69,  56],
    [14, 17, 22, 29,  51,  87,  80,  62],
    [18, 22, 37, 56,  68, 109, 103,  77],
    [24, 35, 55, 64,  81, 104, 113,  92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103,  99]])

def a2dm2(Fbar, beta, rho, sigma, eps):
    # size
    J = Fbar.shape[0]
    I = Fbar.shape[1]

    # initialize
    primal = ibdct(Fbar)
    aux = dual = auxhat = dualhat = numpy.zeros((J, I, 3))
    aold = acur = 1.0

    # filter
    filter = numpy.zeros((J, I))
    filter[0, 0] = 5
    filter[-1, 0] = filter[0, -1] = filter[1, 0] = filter[0, 1] = -1
    filter = numpy.fft.fft2(filter)

    # contraint
    view = skimage.util.view_as_blocks(Fbar, (8, 8))
    l = view - sigma * Q
    u = view + sigma * Q
    l = numpy.reshape(l.swapaxes(1, 2), (J, I))
    u = numpy.reshape(u.swapaxes(1, 2), (J, I))

    for mu in range(4096):
        aold = acur
        dualold = dual

        # update primal
        buf0 = auxhat + dualhat / rho
        buf1 = numpy.hstack((buf0[:, I-1:I, 0], buf0[:, 0:I-1, 0]))
        buf1 = buf1 - buf0[:, :, 0]
        buf2 = numpy.vstack((buf0[J-1:J, :, 1], buf0[0:J-1, :, 1]))
        buf2 = buf2 - buf0[:, :, 1]
        buf3 = ibdct(buf0[:, :, 2])
        buf4 = buf1 + buf2 + buf3
        buf4 = numpy.fft.fft2(buf4)
        buf4 = buf4 / filter
        primal = numpy.real(numpy.fft.ifft2(buf4))

        # update aux
        buf0 = numpy.hstack((primal[:, 1:I], primal[:, 0:1]))
        buf0 = buf0 - primal
        buf1 = numpy.vstack((primal[1:J, :], primal[0:1, :]))
        buf1 = buf1 - primal
        buf2 = bdct(primal)
        buf3 = buf0 - dualhat[:, :, 0] / rho
        buf4 = buf1 - dualhat[:, :, 1] / rho
        buf5 = rho - 1.0 / (numpy.sqrt(buf3 * buf3 + buf4 * buf4) + 1E-12)
        buf5 = numpy.maximum(buf5 / (beta + rho), 0)
        buf6 = buf2 - dualhat[:, :, 2] / rho
        buf6 = rho / (beta + rho) * buf6
        aux[:, :, 0] = buf3 * buf5
        aux[:, :, 1] = buf4 * buf5
        aux[:, :, 2] = numpy.maximum(numpy.minimum(buf6, u), l)

        err0 = (buf0 - aux[:, :, 0]) * (buf0 - aux[:, :, 0])
        err1 = (buf1 - aux[:, :, 1]) * (buf1 - aux[:, :, 1])
        err2 = (buf2 - aux[:, :, 2]) * (buf2 - aux[:, :, 2])
        rmse = numpy.sqrt(numpy.sum(err0 + err1 + err2) / (J + I))

        # update dual
        dual[:, :, 0] = dualhat[:, :, 0] - rho * (buf0 - aux[:, :, 0])
        dual[:, :, 1] = dualhat[:, :, 1] - rho * (buf1 - aux[:, :, 1])
        dual[:, :, 2] = dualhat[:, :, 2] - rho * (buf2 - aux[:, :, 2])

        # update a
        acur = (1.0 + numpy.sqrt(1.0 + 4.0 * aold * aold)) / 2.0

        # update dualhat
        aa = (aold - 1.0) / acur
        dualhat = dual + aa * (dual - dualold)

        # update auxhat
        buf0 = dualhat[:, :, 0]
        buf1 = dualhat[:, :, 1]
        buf2 = 1.0 - 1.0 / (numpy.sqrt(buf0 * buf0 + buf1 * buf1) + 1E-12)
        buf2 = -numpy.maximum(buf2 / beta, 0)
        auxhat[:, :, 0] = buf0 * buf2
        auxhat[:, :, 1] = buf1 * buf2
        auxhat[:, :, 2] = -dualhat[:, :, 2] / beta
        auxhat[:, :, 2] = numpy.maximum(numpy.minimum(auxhat[:, :, 2], u), l)

        if rmse < eps:
            break

    return primal


def bdct(img):
    I = img.shape[1]
    J = img.shape[0]
    view = skimage.util.view_as_blocks(img, (8, 8))
    view = scipy.fftpack.dct(view, axis=2, norm='ortho')
    view = scipy.fftpack.dct(view, axis=3, norm='ortho')
    cff = numpy.reshape(view.swapaxes(1, 2), (J, I))
    return cff


def ibdct(cff):
    I = cff.shape[1]
    J = cff.shape[0]
    view = skimage.util.view_as_blocks(cff, (8, 8))
    view = scipy.fftpack.idct(view, axis=2, norm='ortho')
    view = scipy.fftpack.idct(view, axis=3, norm='ortho')
    img = numpy.reshape(view.swapaxes(1, 2), (J, I))
    return img


if __name__ == '__main__':
    # read
    f = numpy.array(skimage.io.imread('0884.png'), 'double')

    # size
    I = f.shape[1]
    J = f.shape[0]

    # forward dct
    F = bdct(f)

    # quantization
    view = skimage.util.view_as_blocks(F, (8, 8))
    view = numpy.round(view / Q)
    Fdash = numpy.reshape(view.swapaxes(1, 2), (J, I))

    # dequantization
    view = skimage.util.view_as_blocks(Fdash, (8, 8))
    view = view * Q
    Fbar = numpy.reshape(view.swapaxes(1, 2), (J, I))

    # inverse dct
    fbar = ibdct(Fbar)

    # clip
    fbar = numpy.round(fbar)
    fbar = numpy.maximum(fbar, 0)
    fbar = numpy.minimum(fbar, 255)
    fbar = numpy.array(fbar, 'uint8')

    # save
    skimage.io.imsave('dst_jpeg.png', fbar)

    # result (jpeg)
    psnr = skimage.measure.compare_psnr(f, fbar, 255)
    print('jpeg : ' + '{:5.2f}'.format(psnr) + '[dB]')

    # a2dm2
    fbar = a2dm2(Fbar, 0.1, 1, 0.1, 0.1)

    # clip
    fbar = numpy.round(fbar)
    fbar = numpy.maximum(fbar, 0)
    fbar = numpy.minimum(fbar, 255)
    fbar = numpy.array(fbar, 'uint8')

    # save
    skimage.io.imsave('dst_a2dm2.png', fbar)

    # result (a2dm2)
    psnr = skimage.measure.compare_psnr(f, fbar, 255)
    print('a2dm2: ' + '{:5.2f}'.format(psnr) + '[dB]')
