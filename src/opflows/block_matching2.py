import argparse
import os
import itertools
import math
import sys

import numpy as np
import cv2 as cv
from tqdm import tqdm





class EBMA_searcher():
    """
    Estimates the motion between to frame images
     by running an Exhaustive search Block Matching Algorithm (EBMA).
    Minimizes the norm of the Displaced Frame Difference (DFD).
    """

    def __init__(self, N, R, p=1, acc=1):
        """
        :param N: Size of the blocks the image will be cut into, in pixels.
        :param R: Range around the pixel where to search, in pixels.
        :param p: Norm used for the DFD. p=1 => MAD, p=2 => MSE. Default: p=1.
        :param acc: 1: Integer-Pel Accuracy (no interpolation),
                    2: Half-Integer Accuracy (Bilinear interpolation)
        """

        self.N = N
        self.R = R
        self.p = p
        self.acc = acc
        
    def get_original_size(flow,frame_size):
        pass

    def run(self, target_frame, anchor_frame):
        """
        Run!
        :param anchor_frame: Image that will be predicted.
        :param target_frame: Image that will be used to predict the target frame.
        :return: A tuple consisting of the predicted image and the motion field.
        """

        acc = self.acc
        height = anchor_frame.shape[0]
        width = anchor_frame.shape[1]
        N = self.N
        R = self.R
        p = self.p

        # interpolate original images if half-pel accuracy is selected
        if acc == 1:
            pass
        elif acc == 2:
            target_frame = cv.resize(target_frame, dsize=(width * 2, height * 2))
        else:
            raise ValueError('pixel accuracy should be 1 or 2. Got %s instead.' % acc)

        # predicted frame. anchor_frame is predicted from target_frame
        predicted_frame = np.empty((height, width), dtype=np.uint8)

        # motion field consisting in the displacement of each block in vertical and horizontal
        motion_field = np.empty((int(height / N), int(width / N), 2))

        # loop through every NxN block in the target image
        for (blk_row, blk_col) in tqdm(itertools.product(range(0, height - (N - 1), N),
                                                    range(0, width - (N - 1), N))):

            # block whose match will be searched in the anchor frame
            blk = anchor_frame[blk_row:blk_row + N, blk_col:blk_col + N]

            # minimum norm of the DFD norm found so far
            dfd_n_min = np.infty

            # search which block in a surrounding RxR region minimizes the norm of the DFD. Blocks overlap.
            for (r_col, r_row) in itertools.product(range(-R, (R + N)),
                                                    range(-R, (R + N))):
                # candidate block upper left vertex and lower right vertex position as (row, col)
                up_l_candidate_blk = ((blk_row + r_row) * acc, (blk_col + r_col) * acc)
                low_r_candidate_blk = ((blk_row + r_row + N - 1) * acc, (blk_col + r_col + N - 1) * acc)

                # don't search outside the anchor frame. This lowers the computational cost
                if up_l_candidate_blk[0] < 0 or up_l_candidate_blk[1] < 0 or \
                                low_r_candidate_blk[0] > height * acc - 1 or low_r_candidate_blk[1] > width * acc - 1:
                    continue

                # the candidate block may fall outside the anchor frame
                candidate_blk = subarray(target_frame, up_l_candidate_blk, low_r_candidate_blk)[::acc, ::acc]
                assert candidate_blk.shape == (N, N)

                dfd = np.array(candidate_blk, dtype=np.float16) - np.array(blk, dtype=np.float16)

                candidate_dfd_norm = np.linalg.norm(dfd, ord=p)

                # a better matching block has been found. Save it and its displacement
                if candidate_dfd_norm < dfd_n_min:
                    dfd_n_min = candidate_dfd_norm
                    matching_blk = candidate_blk
                    dy = r_col
                    dx = r_row

            # construct the predicted image with the block that matches this block
            predicted_frame[blk_row:blk_row + N, blk_col:blk_col + N] = matching_blk

            # print( str((blk_row / N, blk_col / N)) + '--- Displacement: ' + str((dx, dy)))

            # displacement of this block in each direction
            motion_field[int(blk_row / N), int(blk_col / N), 1] = dx
            motion_field[int(blk_row / N), int(blk_col / N), 0] = dy

        return predicted_frame, motion_field



def positive_integer(number):
    """
    Convert a number to an positive integer if possible.
    :param number: The number to be converted to a positive integer.
    :return: The positive integer
    :raise: argparse.ArgumentTypeError if not able to do the conversion
    """
    try:
        integ = int(number)
        if integ >= 0:
            return integ
        else:
            raise argparse.ArgumentTypeError('%s is not a positive integer' % number)
    except ValueError:
        raise argparse.ArgumentTypeError('%s is not a positive integer' % number)


def subarray(array, param1, param2):
    """
    Return a subarray containing the pixels delimited by the pixels between upper_left_pix and lower_right.
    If asked for pixels outside the image boundary, such pixels have value 0.
    """
    upper_left_pix_row, upper_left_pix_col = param1
    lower_right_pix_row, lower_right_pix_col = param2
    
    if upper_left_pix_row > lower_right_pix_row or upper_left_pix_col > lower_right_pix_col:
        raise ValueError('coordinates of the subarray should correspond to a meaningful rectangle')

    orig_array = np.array(array)

    num_rows = lower_right_pix_row - upper_left_pix_row + 1
    num_cols = lower_right_pix_col - upper_left_pix_col + 1

    subarr = np.zeros((num_rows, num_cols), dtype=orig_array.dtype)

    # zoomed outside the original image
    if lower_right_pix_col < 0 or lower_right_pix_row < 0 or \
                    upper_left_pix_col > orig_array.shape[1] - 1 or upper_left_pix_row > orig_array.shape[0] - 1:
        return subarr

    # region of the original image that is inside the desired region
    # (i = col, j=row)
    # _________________________________
    # |                                | original image
    # |   _____________________________|____
    # |   |(j_o_1, i_o_1)              |    |
    # |   |             (j_o_2, i_o_2) |    |
    # |___|____________________________|    |
    # |                                 |
    # |_________________________________|  sliced final image

    if upper_left_pix_col < 0:
        i_o_1 = 0
    else:
        i_o_1 = upper_left_pix_col
    if upper_left_pix_row < 0:
        j_o_1 = 0
    else:
        j_o_1 = upper_left_pix_row

    if lower_right_pix_col > orig_array.shape[1] - 1:
        i_o_2 = orig_array.shape[1] - 1
    else:
        i_o_2 = lower_right_pix_col
    if lower_right_pix_row > orig_array.shape[0] - 1:
        j_o_2 = orig_array.shape[0] - 1
    else:
        j_o_2 = lower_right_pix_row


    # region of the final image that is inside the original image, and whose content will be taken from the orig im
    # (i = col, j=row)
    # _________________________________
    # |                                | original image
    # |   _____________________________|____
    # |   |(j_f_1, i_f_1)              |    |
    # |   |             (j_f_2, i_f_2) |    |
    # |___|____________________________|    |
    #     |                                 |
    #     |_________________________________|  sliced final image

    if upper_left_pix_col < 0:
        i_f_1 = -upper_left_pix_col
    else:
        i_f_1 = 0
    if upper_left_pix_row < 0:
        j_f_1 = -upper_left_pix_row
    else:
        j_f_1 = 0

    if lower_right_pix_col > orig_array.shape[1] - 1:
        i_f_2 = (orig_array.shape[1] - 1) - upper_left_pix_col
    else:
        i_f_2 = num_cols - 1
    if lower_right_pix_row > orig_array.shape[0] - 1:
        j_f_2 = (orig_array.shape[0] - 1) - upper_left_pix_row
    else:
        j_f_2 = num_rows - 1

    subarr[j_f_1:j_f_2 + 1, i_f_1:i_f_2 + 1] = orig_array[j_o_1:j_o_2 + 1, i_o_1:i_o_2 + 1]

    return subarr