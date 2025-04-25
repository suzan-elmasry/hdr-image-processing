import numpy as np
import os
import skimage
import time
import matplotlib.pyplot as plt

def rescale(img, window=(0, 1)):
    """
       This function is used to rescale the intensity of the image from [Imin, Imax] to window=(min, max).
    """
    a = np.min(img)
    b = np.max(img)
    if a == b:
        if a == 0:
            temp = img
        else:
            temp = img / a
    else:
        temp = (img - a) / (b - a) * (window[1] - window[0]) + window[0]
    return temp

def crop_image(img, crop_size):
    """
       This function is used to central crop image.
    """
    h, w = img.shape[:2]
    crop_h = min(crop_size[0], h)
    crop_w = min(crop_size[1], w)
    start_h = (h - crop_h) // 2
    start_w = (w - crop_w) // 2
    return img[start_h:start_h + crop_h, start_w:start_w + crop_w]

def hist_equalize(photo, numtiles=(8, 8)):
    """
       This function is used to applied Contrast Limited Adaptive Histogram Equalization (CLAHE) to an image.
    """
    assert photo.shape[0] % numtiles[0] == 0 and photo.shape[1] % numtiles[1] == 0
    img_range = np.array([0, 1])
    tile_size = (photo.shape[0] // numtiles[0], photo.shape[1] // numtiles[1])
    num_bins = 256
    norm_clip_limit = 0.01

    num_pixel_in_tile = np.prod(tile_size)
    min_clip_limit = np.ceil(np.float32(num_pixel_in_tile) / num_bins)
    clip_limit = min_clip_limit + np.round(norm_clip_limit * (num_pixel_in_tile - min_clip_limit))
    print(clip_limit)
    tile_mappings = []
    image_row = 0
    print('make tile mappings')

    for tile_row in range(numtiles[0]):
        tile_mappings.append([])
        image_col = 0
        for tile_col in range(numtiles[1]):
            print('tile ({}, {}):'.format(tile_row, tile_col), end=',')
            tile = photo[image_row:(image_row + tile_size[0]), image_col:(image_col + tile_size[1])]
            tile_hist = imhist(tile, num_bins, img_range[1])
            tile_hist = clip_histogram(tile_hist, clip_limit, num_bins)
            tile_mapping = make_mapping(tile_hist, img_range, num_pixel_in_tile)
            tile_mappings[-1].append(tile_mapping)

            image_col += tile_size[1]
        image_row += tile_size[0]

    # tile_mappings = maketile_mapping(photo, numtiles, tile_size, img_range, img_range)

    out = make_clahe_image(photo, tile_mappings, numtiles, tile_size, img_range)
    return out

def imhist(tile, num_bins, top):
    s = (num_bins - 1.) / top   # scale factor
    tile_scaled = np.floor(tile * s + .5)
    hist = np.zeros(num_bins, dtype=np.int32)
    for i in range(num_bins):
        hist[i] = np.sum(tile_scaled == i)
    return hist

def clip_histogram(img_hist, clip_limit, num_bins):
    total_excess = np.sum(np.maximum(img_hist - clip_limit, 0))
    avg_bin_incr = np.floor(total_excess / num_bins)
    upper_limit = clip_limit - avg_bin_incr

    for k in range(num_bins):
        if img_hist[k] > clip_limit:
            img_hist[k] = clip_limit
        else:
            if img_hist[k] > upper_limit:
                total_excess -= clip_limit - img_hist[k]
                img_hist[k] = clip_limit
            else:
                total_excess -= avg_bin_incr
                img_hist[k] += avg_bin_incr

    k = 0
    print('total excess={}'.format(total_excess), end=';')
    while total_excess != 0:
        step_size = max(int(np.floor(num_bins / total_excess)), 1)
        for m in range(k, num_bins, step_size):
            if img_hist[m] < clip_limit:
                img_hist[m] += 1
                total_excess -= 1
            if total_excess == 0:
                break
        k += 1
        if k == num_bins:
            k = 0
    return img_hist

def make_mapping(img_hist, selected_range, num_pixel_in_tile):
    high_sum = np.cumsum(img_hist)
    val_spread = selected_range[1] - selected_range[0]

    scale = val_spread / num_pixel_in_tile
    mapping = np.minimum(selected_range[0] + high_sum * scale, selected_range[1])
    return mapping

def make_clahe_image(photo, tile_mappings, numtiles, tile_size, selected_range, num_bins=256):
    assert num_bins > 1
    print('make clahe image')
    Ic = np.zeros_like(photo)

    bin_step = 1. / (num_bins - 1)
    start = np.ceil(selected_range[0] / bin_step)
    stop = np.floor(selected_range[1] / bin_step)
    aLut = np.arange(0, 1 + 1e-10, 1.0 / (stop - start))

    imgtile_row = 0
    for k in range(numtiles[0] + 1):
        if k == 0:  # edge case: top row
            imgtile_num_rows = tile_size[0] // 2
            maptile_rows = (0, 0)
        elif k == numtiles[0]:
            imgtile_num_rows = tile_size[0] // 2
            maptile_rows = (numtiles[0] - 1, numtiles[0] - 1)
        else:
            imgtile_num_rows = tile_size[0]
            maptile_rows = (k - 1, k)

        imgtile_col = 0
        for l in range(numtiles[1] + 1):
            print('tile ({}, {})'.format(k, l))
            if l == 0:
                imgtile_num_cols = tile_size[1] // 2
                maptile_cols = (0, 0)
            elif l == numtiles[1]:
                imgtile_num_cols = tile_size[1] // 2
                maptile_cols = (numtiles[1] - 1, numtiles[1] - 1)
            else:
                imgtile_num_cols = tile_size[1]
                maptile_cols = (l - 1, l)

            ul_maptile = tile_mappings[maptile_rows[0]][maptile_cols[0]]
            ur_maptile = tile_mappings[maptile_rows[0]][maptile_cols[1]]
            bl_maptile = tile_mappings[maptile_rows[1]][maptile_cols[0]]
            br_maptile = tile_mappings[maptile_rows[1]][maptile_cols[1]]

            norm_factor = imgtile_num_rows * imgtile_num_cols

            imgpxl_vals = grayxform(photo[imgtile_row:(imgtile_row + imgtile_num_rows), imgtile_col:(imgtile_col + imgtile_num_cols)], aLut)

            row_w = np.tile(np.expand_dims(np.arange(imgtile_num_rows), axis=1), [1, imgtile_num_cols])
            col_w = np.tile(np.expand_dims(np.arange(imgtile_num_cols), axis=0), [imgtile_num_rows, 1])
            row_rev_w = np.tile(np.expand_dims(np.arange(imgtile_num_rows, 0, -1), axis=1), [1, imgtile_num_cols])
            col_rev_w = np.tile(np.expand_dims(np.arange(imgtile_num_cols, 0, -1), axis=0), [imgtile_num_rows, 1])

            Ic[imgtile_row:(imgtile_row + imgtile_num_rows), imgtile_col:(imgtile_col + imgtile_num_cols)] = (row_rev_w * (col_rev_w * grayxform(imgpxl_vals, ul_maptile) + col_w * grayxform(imgpxl_vals, ur_maptile)) + row_w * (col_rev_w * grayxform(imgpxl_vals, bl_maptile) + col_w * grayxform(imgpxl_vals, br_maptile))) / norm_factor

            imgtile_col += imgtile_num_cols

        imgtile_row += imgtile_num_rows
    return Ic


def grayxform(photo, aLut):
    max_idx = len(aLut) - 1
    val = np.copy(photo)
    val[val < 0] = 0
    val[val > 1] = 1
    indexes = np.int32(val * max_idx + 0.5)
    return aLut[indexes]

def tonemap(E, l_remap=(0, 1), saturation=1., numtiles=(4, 4)):
    if E.shape[0] % numtiles[0] != 0 or E.shape[1] % numtiles[1] != 0:
        E = crop_image(E, (E.shape[0] // numtiles[0] * numtiles[0], E.shape[1] // numtiles[1] * numtiles[1]))
    l2E, has_nonzero = lognormal(E)
    if has_nonzero:
        I = tone_operator(l2E, l_remap, saturation, numtiles)
    else:
        I = l2E
    # clip
    I[I < 0] = 0
    I[1 < I] = 1
    return np.uint8(I * 255.)


def lognormal(E):
    """
        log2(E)
    """
    mask = (E != 0)

    if np.any(mask):
        min_nonzero = np.min(E[mask])
        E[np.logical_not(mask)] = min_nonzero
        l2E = rescale(np.log2(E))
        has_nonzero = True

    else:
        l2E = np.zeros_like(E)
        has_nonzero = False

    return l2E, has_nonzero

def tone_operator(l2E, l_remap, saturation, numtiles):
    """
        The main algorithm is CLAHE: contrast limited adaptive histogram equalization
    """
    # lab = srgb2lab(l2E)
    lab = skimage.color.rgb2lab(l2E)
    lab[:,:,0] = rescale(lab[:,:,0])
#    lab[:, :, 0] /= 100
    lab[:, :, 0] = hist_equalize(lab[:, :, 0], numtiles)
    lab[:, :, 0] = imadjust(lab[:, :, 0], range_in=l_remap, range_out=(0, 1), gamma=1.5) * 100
    lab[:, :, 1:] = lab[:, :, 1:] * saturation
    # img = lab2srgb(lab)
    I = skimage.color.lab2rgb(lab)
    return I

def imadjust(I, range_in=None, range_out=(0, 1), gamma=1):
    """
        remap img from range_in to range_out
    """
    if range_in is None:
        range_in = (np.min(I), np.max(I))
    out = (I - range_in[0]) / (range_in[1] - range_in[0])
    out = out**gamma
    out = out * (range_out[1] - range_out[0]) + range_out[0]
    return out
def w_mid(z):
    return 1 - np.abs(z - 128.0) / 130.0

def gsolve(points, ln_te, lmd, weight_fcn=w_mid):
    N, M = points.shape
    nlevels = 256
    A = np.zeros((N * M + nlevels - 1, nlevels + N))
    b = np.zeros(A.shape[0])
    print('system size:{}'.format(A.shape))
    t_s = time.time()

    k = 0
    for i in range(N):
        for j in range(M):
            wij = weight_fcn(points[i, j])
            A[k, int(points[i, j])] = wij
            A[k, nlevels + i] = -wij
            b[k] = wij * ln_te[j]
            k += 1

    A[k, 128] = 1   # Fix the curve by setting its middle value to 0
    k += 1

    for i in range(nlevels-2):
        wi = weight_fcn(i)
        A[k, i] = lmd * wi
        A[k, i+1] = -2 * lmd * wi
        A[k, i+2] = lmd * wi
        k += 1

    x, _, _, _ = np.linalg.lstsq(A, b)
    g = x[:nlevels]
    lnE = x[nlevels:]
    t_e = time.time()
    return g, lnE


def hdr_recon(g, photo, ln_te, weight_fcn=w_mid):
    lnE = np.zeros(photo[0].shape)
    ln_te_mat = np.array([np.tile(ln_te[i], photo.shape[1:-1]) for i in range(photo.shape[0])])
    for ch in range(3):
        weighted_sum = np.sum(weight_fcn(photo[:, :, :, ch]) * (g[ch][photo[:, :, :, ch]] - ln_te_mat), axis=0)
        weight_sum = np.sum(weight_fcn(photo[:, :, :, ch]), axis=0)
        lnE[:,:,ch] = weighted_sum / weight_sum
    plt.figure()
    plt.imshow(np.exp(lnE))
    plt.title('HDR Radiance Map')
    plt.axis('off')
    plt.savefig('recovered HDR image.png')
    plt.show()
    return lnE

def readImagesAndTimes():
    times = np.array([ 1/250, 1/100, 1/20, 1/8, 1/4, 1.0, 2, 3], dtype=np.float32)
    filenames = [f"image\DSCF28{i}.jpg"for i in range(38, 46)]
    images = []
    for filename in filenames:
        im = skimage.io.imread(filename)
        images.append(im)
    return images, times

def uniform_grid_sample(imagestack, N):
    M = len(imagestack)
    m, n, _ = imagestack[0].shape
    # sample ratio
    Rx = int((m - 20) / (N - 1))
    Ry = int((n - 20) / (N - 1))
    points = np.zeros((N*N, M, 3))
    print(N * N)
    for ch in range(3):
        for k in range(M):
            points[:,k,ch] = imagestack[k][10:-1:Rx, 10:-1:Ry, ch].ravel()
    return points

def random_sample(imagestack, k=12):
    M = len(imagestack)
    N = int(np.ceil(np.sqrt( (256 * k - 255) / (M - k) )))
    print('point:',N*N,',',M)
    m, n, _ = imagestack[0].shape
    indexes = np.arange(m * n)
    np.random.shuffle(indexes)
    idx_selected = indexes[:N*N]
    points = np.zeros((N*N, M, 3))
    for ch in range(3):
        for k in range(M):
            points[:,k,ch] = imagestack[k][:,:,ch].ravel()[idx_selected]
    return points

def align_two_images(R, T, max_size = 512 ):
    """
    Align two images
    """
    Rcrop = crop_image(R, (max_size, max_size))
    Tcrop = crop_image(T, (max_size, max_size))
    rgb2gray = lambda x: (0.2125 * x[:, :, 0] + 0.7154 * x[:, :, 1] + 0.0721 * x[:, :, 2])
    if len(R.shape) == 3 and R.shape[2] == 3:
        Rg = rgb2gray(Rcrop)
        Tg = rgb2gray(Tcrop)
    elif len(R.shape) == 2:
        Rg = R
        Tg = T
    else:
        raise RuntimeError("Invalid image size!")

    # FFT
    Fr = np.fft.fft2(Rg)
    Ft = np.fft.fft2(Tg)
    Fc = Fr * np.conj(Ft)
    Rc = Fc / np.abs(Fc)
    r = np.fft.ifft2(Rc)
    # get the peak
    max_r = np.max(r)
    max_index = np.argmax(r)
    shift = list(np.unravel_index(max_index, r.shape))
    for i in range(2):
        if r.shape[i] / 2 <= shift[i]:
            shift[i] -= r.shape[i]
    Rt = translate(T, shift)
    return Rt, shift

def translate(T, shift):
    Rt = np.roll(T, shift=shift[0], axis=0)
    Rt = np.roll(T, shift=shift[1], axis=1)
    return Rt

def process(save_name):
    save_path = os.path.join(os.path.join('../output'), save_name + '.jpg')
    if False:
        print('{} already processed'.format(save_name))
    else:
        print('load images')
        imagestack, exposure_times = readImagesAndTimes()
        index = [i[0] for i in sorted(enumerate(exposure_times), key=lambda x:x[1])]
        exposure_times = [exposure_times[i] for i in index]
        imagestack = [imagestack[i] for i in index]
        print(exposure_times)

        # alignment
        print('align images')
        is_aligned = [imagestack[0]]
        shifts = []
        for i in range(1, len(imagestack)):
            print('align image {}'.format(i))
            Treg, shift = align_two_images(is_aligned[i-1], imagestack[i])
            #is_aligned.append(Treg)
            is_aligned.append(imagestack[i])
            shifts.append(shift)
        print(shifts)
        shifts = np.array(shifts)
        margin = np.max(shifts, axis=0)
        assert np.all(margin < 20)
        print('margin:{}'.format(margin))
        print('new image size:{}'.format(is_aligned[0].shape))

        # sample points
        points = uniform_grid_sample(imagestack, N=64)
        # points = random_sample(imagestack)
        I = np.array(is_aligned)

        # recon
        print('estimate exposure')
        lmd = 2
        ln_te = np.log(exposure_times)
        gs = [gsolve(points[:,:,i], ln_te, lmd)[0] for i in range(3)]

        lnE = hdr_recon(gs, I, ln_te)
        E = np.exp(lnE)

        plt.figure(figsize=(10, 7))
        channel_names = ['Red Channel', 'Green Channel', 'Blue Channel']
        colors = ['r', 'g', 'b']  # Red, green, blue channel
        for i, g in enumerate(gs):
            pixel_values = np.arange(len(g))  # range of pixels
            plt.plot(pixel_values, g, color=colors[i], label=channel_names[i] + ' Response')
        plt.title('Camera Response Curves for RGB Channels')
        plt.xlabel('Pixel Value')
        plt.legend()
        plt.grid(True)
        plt.savefig('Camera Response Curves.png')
        plt.show()

        # param
        print('tone mapping')
        l_remap = (0, 1)
        saturation = 2.
        numtiles = (4, 4)
        I = tonemap(E, l_remap=l_remap, saturation=saturation, numtiles=numtiles)
        plt.imshow(I)
        plt.show()
        skimage.io.imsave(save_path, I)

if __name__ == '__main__':
    if not os.path.isdir('../output'):
        os.mkdir('../output')
    save_name = 'tone-mapped image 3, 64'
    process(save_name)