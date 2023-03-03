import dlib
import os
import numpy as np
import cv2
from skimage.transform import PiecewiseAffineTransform, warp


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(os.path.split(os.path.realpath(__file__))[0] + '/dlib_source/lms.dat')


def resize_by_max(image, max_side=512, force=False):
    h, w = image.shape[:2]
    if max(h, w) < max_side and not force:
        return image
    ratio = max(h, w) / max_side

    w = int(w / ratio + 0.5)
    h = int(h / ratio + 0.5)
    return cv2.resize(image, (w, h))


def face(image: np.ndarray):
    '''
    :param image: BGR
    :return:
    '''
    image = np.asarray(image)
    h, w = image.shape[:2]
    image = resize_by_max(image, 361)
    actual_h, actual_w = image.shape[:2]
    faces_on_small = detector(image, 1)
    faces = dlib.rectangles()
    for face in faces_on_small:
        faces.append(
            dlib.rectangle(
                int(face.left() / actual_w * w + 0.5),
                int(face.top() / actual_h * h + 0.5),
                int(face.right() / actual_w * w + 0.5),
                int(face.bottom() / actual_h * h + 0.5)
            )
        )
    return faces


def landmark(image: np.ndarray):
    '''
    :param image: BGR
    :return:
    '''
    rect = face(image)
    if len(rect) == 0:
        return None
    shape = predictor(image, rect[0]).parts()
    return np.float32([[p.x, p.y] for p in shape])


def rebound_box(mask, mask_face):
    index = np.nonzero(mask)
    x_index = index[0]
    y_index = index[1]

    mask_temp = mask.copy()
    mask_temp[min(x_index) - 5:max(x_index) + 6, min(y_index) - 5:max(y_index) + 6] = \
        mask_face[min(x_index) - 5:max(x_index) + 6, min(y_index) - 5:max(y_index) + 6]

    return mask_temp


def makeup_regin(mask):
    # 7上嘴唇 9下嘴唇
    mask_lip = (mask == 9).astype(np.float) + (mask == 7).astype(np.float)
    # 1，6，13脸部皮肤
    mask_skin = (mask == 1).astype(np.float) + (mask == 6).astype(np.float) + (mask == 13).astype(np.float)
    # # 4左眼 5右眼
    # mask_eye_left = (mask == 4).astype(np.float)
    # mask_eye_right = (mask == 5).astype(np.float)
    #
    # mask_face = (mask == 1).astype(np.float) + (mask == 6).astype(np.float)
    # mask_eye_left = rebound_box(mask_eye_left, mask_face)
    # mask_eye_right = rebound_box(mask_eye_right, mask_face)
    # 脖子 耳朵
    mask_bz = (mask == 10).astype(np.float) + (mask == 3).astype(np.float) + (mask == 5).astype(np.float)

    return mask_lip + mask_skin + mask_bz # mask_eye_left + mask_eye_right


# 直方图匹配
def hist_match_func(source, reference):
    """
    Adjust the pixel values of images such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        reference: np.ndarray
            Reference image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """
    source = np.expand_dims(source, axis=0)
    reference = np.expand_dims(reference, axis=0)

    oldshape = source.shape
    batch_size = oldshape[0]
    source = np.array(source, dtype=np.uint8)
    reference = np.array(reference, dtype=np.uint8)
    # get the set of unique pixel values and their corresponding indices and
    # counts
    result = np.zeros(oldshape, dtype=np.uint8)
    for i in range(batch_size):
        for c in range(3):
            s = source[i, ..., c].ravel()
            r = reference[i, ..., c].ravel()

            s_values, bin_idx, s_counts = np.unique(s, return_inverse=True, return_counts=True)
            r_values, r_counts = np.unique(r, return_counts=True)

            if (len(s_counts) == 1 or len(r_counts) == 1):
                continue
            # take the cumsum of the counts and normalize by the number of pixels to
            # get the empirical cumulative distribution functions for the source and
            # template images (maps pixel value --> quantile)
            s_quantiles = np.cumsum(s_counts[1:]).astype(np.float64)
            s_quantiles /= s_quantiles[-1]
            r_quantiles = np.cumsum(r_counts[1:]).astype(np.float64)
            r_quantiles /= r_quantiles[-1]
            r_values = r_values[1:]

            # interpolate linearly to find the pixel values in the template image
            # that correspond most closely to the quantiles in the source image
            interp_value = np.zeros_like(s_values, dtype=np.float32)
            interp_r_values = np.interp(s_quantiles, r_quantiles, r_values)
            interp_value[1:] = interp_r_values
            result[i, ..., c] = interp_value[bin_idx].reshape(oldshape[1:3])
    result = np.array(result, dtype=np.float32)
    return result[0]


def warp_hismatch(from_img, to_img, from_pts, to_pts, from_seg, to_seg):
    shape = from_img.shape[:2]
    m, _ = cv2.estimateAffinePartial2D(from_pts, to_pts, method=cv2.RANSAC)
    warp_img = cv2.warpAffine(from_img, m, dsize=shape)

    from_mask = makeup_regin(from_seg)
    to_mask = makeup_regin(to_seg)[:, :, np.newaxis]
    warp_mask = cv2.warpAffine(from_mask, m, dsize=shape)[:, :, np.newaxis]

    unchange = warp_img * np.repeat(1 - warp_mask, 3, 2)
    result = hist_match_func(warp_img * np.repeat(warp_mask, 3, 2), to_img * np.repeat(to_mask, 3, 2))
    return result + unchange


def pseudo_paired():
    in_root = '/home/shengdewu/data/makeup.data/MT-Dataset'
    out_root = '/home/shengdewu/data/makeup.data/MT-Dataset/images/warp'

    makeup_names = os.listdir(f'{in_root}/images/makeup')
    nonmakeup_names = os.listdir(f'{in_root}/images/non-makeup')

    for makeup_name in makeup_names:
        for nonmakeup_name in nonmakeup_names:
            makeup_img = cv2.imread(f'{in_root}/images/makeup/{makeup_name}')
            nonmakeup_img = cv2.imread(f'{in_root}/images/non-makeup/{nonmakeup_name}')
            makeup_seg_img = cv2.imread(f'{in_root}/segs/makeup/{makeup_name}', cv2.IMREAD_GRAYSCALE)
            nonmakeup_seg_img = cv2.imread(f'{in_root}/segs/non-makeup/{nonmakeup_name}', cv2.IMREAD_GRAYSCALE)
            makeup_seg_img = cv2.resize(makeup_seg_img, dsize=(makeup_img.shape[1], makeup_img.shape[0]), interpolation=cv2.INTER_NEAREST)
            nonmakeup_seg_img = cv2.resize(nonmakeup_seg_img, dsize=(nonmakeup_img.shape[1], nonmakeup_img.shape[0]), interpolation=cv2.INTER_NEAREST)

            make_pts = landmark(makeup_img)
            nonmake_pts = landmark(nonmakeup_img)

            nonmake2make = makeup_name.split('.')[0] + '_' + nonmakeup_name.split('.')[0] + '.jpg'
            make2nonmake = nonmakeup_name.split('.')[0] + '_' + makeup_name.split('.')[0] + '.jpg'

            if make_pts is None or nonmake_pts is None:
                # cv2.imwrite(f'test/warp/{nonmake2make}', np.concatenate([makeup_img, nonmakeup_img], axis=1))
                print('is empty')
                continue

            nonmakeup2makeup = warp_hismatch(nonmakeup_img, makeup_img, nonmake_pts, make_pts, nonmakeup_seg_img, makeup_seg_img)
            makeup2nonmakeup = warp_hismatch(makeup_img, nonmakeup_img, make_pts, nonmake_pts, makeup_seg_img, nonmakeup_seg_img)

            for i, pts in enumerate(make_pts):
                cv2.putText(makeup_img, f'{i}', (int(pts[0]), int(pts[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            for i, pts in enumerate(nonmake_pts):
                cv2.putText(nonmakeup_img, f'{i}', (int(pts[0]), int(pts[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            img = np.concatenate([makeup_img, makeup2nonmakeup, nonmakeup_img, nonmakeup2makeup], axis=1)
            cv2.imwrite(f'test/warp/{nonmake2make}', img)


pseudo_paired()


