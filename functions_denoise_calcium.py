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


if __name__ == "__main__":
    import ffmpeg
    from glob import glob
    import os
    import re


    def load_avi(file_name):

        """Load an avi video from the wirefree miniscope (based on minian code)"""
        # get the file info
        info = ffmpeg.probe(file_name)
        video_info = next(s for s in info["streams"] if s["codec_type"] == "video")
        w = int(video_info["width"])
        h = int(video_info["height"])
        f = int(video_info["nb_frames"])
        # load the file
        out_bytes, err = (
            ffmpeg.input(file_name)
            .video.output("pipe:", format="rawvideo", pix_fmt="gray")
            .run(capture_stdout=True)
        )
        stack_out = np.frombuffer(out_bytes, np.uint8).reshape(f, h, w).copy()
        # get rid of the 0 frames
        keep_idx = np.sum(np.sum(stack_out, axis=2), axis=1) != 0
        stack_out = stack_out[keep_idx, :, :]
        return stack_out


    def concatenate_wirefree_video(filenames, processing_path=None):
        """Concatenate the videos from a single recording into a single tif file"""
        # based on https://stackoverflow.com/questions/47182125/how-to-combine-tif-stacks-in-python

        # read the first stack on the list
        im_1 = load_avi(filenames[0])

        # save the file name and the number of frames
        frames_list = [[filenames[0], im_1.shape[0]]]
        # assemble the output path
        if processing_path is not None:
            # get the basename
            base_name = os.path.basename(filenames[0])
            out_path_tif = os.path.join(processing_path, base_name.replace('.avi', '_CAT.tif'))
            out_path_log = os.path.join(processing_path, base_name.replace('.avi', '_CAT.csv'))
        else:
            out_path_tif = filenames[0].replace('.avi', '_CAT.tif')
            out_path_log = filenames[0].replace('.avi', '_CAT.csv')
        # run through the remaining files
        for i in range(1, len(filenames)):
            # load the next file
            im_n = load_avi(filenames[i])

            # concatenate it to the previous one
            im_1 = np.concatenate((im_1, im_n))
            # save the file name and the number of frames
            frames_list.append([filenames[i], im_n.shape[0]])
        # scale the output to max and turn into uint8 (for MiniAn)
        max_value = np.max(im_1)
        for idx, frames in enumerate(im_1):
            im_1[idx, :, :] = ((frames / max_value) * 255).astype('uint8')
        # save the final stack
        io.imsave(out_path_tif, im_1, plugin='tifffile', bigtiff=True)
        # save the info about the files
        # frames_list = pd.DataFrame(frames_list, columns=['filename', 'frame_number'])
        # frames_list.to_csv(out_path_log)

        return out_path_tif, out_path_log, im_1

    base_path = r"D:\test_041423"
    sub_dirs = os.listdir(base_path)
    sub_dirs = [os.path.join(base_path, d) for d in sub_dirs]

    for item_path in sub_dirs:

        if not os.path.isfile(item_path):
            videos = glob(os.path.join(item_path, "*.avi"))
            videos.sort(key=lambda f: int(re.sub(r'\D', '', f)))
            video_path, _, _ = concatenate_wirefree_video(videos, processing_path=item_path)
        else:
            video_path = item_path

        out_path = video_path.replace('.tif', '_denoised.tif')

        print(f"Denoising {video_path} ...")
        denoised_stack = denoise_stack(video_path)

        # Save the denoised stack
        io.imsave(out_path, denoised_stack, plugin="tifffile", bigtiff=True)

    print("Done!\n")

