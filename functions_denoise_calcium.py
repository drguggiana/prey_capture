import numpy as np
import cv2
from scipy.signal import butter, filtfilt
from skimage import io


def get_sum_fft(stack, frame_skip=10, apply_vignette=False):
    """ Run a 2D Gaussian kernel over the stack and and compute the FFT sum"""
    rows, cols = stack.shape[-2:] 
    
    # Generate Gaussian Kernel
    X_resultant_kernel = cv2.getGaussianKernel(cols, cols/4) 
    Y_resultant_kernel = cv2.getGaussianKernel(rows, rows/4) 
    resultant_kernel = X_resultant_kernel * Y_resultant_kernel.T 

    # Create mask
    mask = 255 * resultant_kernel / np.linalg.norm(resultant_kernel)
    
    if not apply_vignette:
        mask = mask * 0 + 1
        
    mask = np.expand_dims(mask, axis=0)

    masked_stack = stack[::frame_skip, :] * mask

    dft = np.fft.fftn(np.float32(masked_stack), axes=[1, 2])
    dft_shift = np.fft.fftshift(dft, axes=[1, 2])
    sum_fft = np.sum(np.absolute(dft_shift), axis=0)
    return sum_fft


def get_mask_fft(fft, good_radius=2000, notch_hw=3, center_half_height=20):
    """Modify FFT using a circle mask around center"""
    
    rows, cols = fft.shape[-2:]
    crow, ccol = int(rows/2), int(cols/2)

    mask_fft = np.zeros((rows, cols), np.float32)
    cv2.circle(mask_fft, (crow, ccol), good_radius, 1, thickness=-1)

    mask_fft[(crow + center_half_height):, (ccol - notch_hw):(ccol + notch_hw)] = 0
    mask_fft[:(crow - center_half_height), (ccol - notch_hw):(ccol + notch_hw)] = 0
    return mask_fft


def apply_spatial_filter(stack, fft_mask):
    """Apply the FFT mask to the data and filter"""

    # Need to go frame by frame to avoid memory overflow
    stack_back = np.zeros(stack.shape, dtype=np.uint8)
    for i, frame in enumerate(stack):
        dft = np.fft.fftn(frame)
        dft_shift = np.fft.fftshift(dft)
        fshift = dft_shift * fft_mask
        f_ishift = np.fft.ifftshift(fshift)
        frame_back = np.fft.ifftn(f_ishift)
        frame_back = np.absolute(frame_back)
        frame_back[frame_back > 255] = 255
        stack_back[i, :] = frame_back.astype(np.uint8)

    return stack_back


def apply_lpf(stack: np.array, fs=20, cutoff=4, order=6):
    """Construct a lowpass filter to remove low-frequency fluorescence changes"""

    b, a = butter(order, cutoff / (0.5 * fs), btype='low', analog=False)
    
    mean_fluor = np.mean(stack, axis=(1, 2))
    mean_filt = filtfilt(b, a, mean_fluor)
    
    filtered_diff = 1 + (mean_filt - mean_fluor)/mean_fluor
    stack_lpf = stack * np.expand_dims(filtered_diff, axis=(1, 2))
    stack_lpf[stack_lpf > 255] = 255
    stack_lpf = stack_lpf.astype(np.uint8)
    return stack_lpf, mean_fluor, mean_filt


def denoise_stack(filename):
    """Runs the denoosing functions"""
    # Load the tif
    stack = io.imread(filename).astype(np.uint8)

    # if it's 2d (i.e 1 frame), expand 1 dimension
    if len(stack.shape) == 2:
        stack = np.expand_dims(stack, 2)
        stack = np.transpose(stack, [2, 0, 1])

    # Run denoising
    sum_fft = get_sum_fft(stack, frame_skip=100)
    mask_fft = get_mask_fft(sum_fft)
    stack_spatial_filt = apply_spatial_filter(stack, mask_fft)
    stack_lowpass, _, _ = apply_lpf(stack_spatial_filt, cutoff=3.5, order=9)

    return stack_lowpass
