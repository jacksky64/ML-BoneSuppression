from io import BytesIO
import argparse
import random 
import cv2
import numpy as np
from matplotlib import pyplot as plt

from albumentations import (
    Compose, ToFloat, FromFloat, RandomRotate90, Flip, HorizontalFlip, OneOf, MotionBlur, MedianBlur, Blur,
    ShiftScaleRotate, OpticalDistortion, GridDistortion, RandomBrightnessContrast,Transpose,IAAAdditiveGaussianNoise,
    HueSaturationValue,GaussNoise,IAAPiecewiseAffine,CLAHE,IAASharpen,IAAEmboss,RandomBrightness
)

def strong_tiff_aug(p=.5):
    return Compose([
        # albumentations supports uint8 and float32 inputs. For the latter, all
        # values must lie in the range [0.0, 1.0]. To apply augmentations, we
        # first use a `ToFloat()` transformation, which will inspect the data
        # type of the input image and convert the image to a float32 ndarray where
        # all values lie in the required range [0.0, 1.0].
        ToFloat(),
        
        # Alternatively, you can specify the maximum possible value for your input
        # and all values will be divided by it instead of using a predefined value
        # for a specific data type.       
        # ToFloat(max_value=65535.0),
        
        # Then we will apply augmentations
        RandomRotate90(),
        Flip(),
        OneOf([
            MotionBlur(p=0.2),
            MedianBlur(blur_limit=3, p=0.1),
            Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        OneOf([
            OpticalDistortion(p=0.3),
            GridDistortion(p=0.1),
        ], p=0.2),
        
        
        # You can convert the augmented image back to its original
        # data type by using `FromFloat`.
        # FromFloat(dtype='uint16'),

        # As in `ToFloat` you can specify a `max_value` argument and all input values
        # will be multiplied by it.
        FromFloat(dtype='uint16', max_value=65535.0),

    ], p=p)

def strong_aug(p=1):
    return Compose([
        ToFloat(),
        RandomRotate90(),
        Flip(),
        Transpose(),
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=0.2),
        OneOf([
            MotionBlur(p=0.2),
            MedianBlur(blur_limit=3, p=0.1),
            Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        OneOf([
            OpticalDistortion(p=0.3),
            GridDistortion(p=0.1),
            IAAPiecewiseAffine(p=0.3),
        ], p=0.2),
        FromFloat(dtype='uint16', max_value=65535.0)
    ], p=p)

def light_aug(p=1):
    return Compose([
        # albumentations supports uint8 and float32 inputs. For the latter, all
        # values must lie in the range [0.0, 1.0]. To apply augmentations, we
        # first use a `ToFloat()` transformation, which will inspect the data
        # type of the input image and convert the image to a float32 ndarray where
        # all values lie in the required range [0.0, 1.0].
        ToFloat(),
        
        RandomBrightness(limit=(0,0.2), p=0.2),

        # Alternatively, you can specify the maximum possible value for your input
        # and all values will be divided by it instead of using a predefined value
        # for a specific data type.       
        # ToFloat(max_value=65535.0),
        
        # Then we will apply augmentations
        HorizontalFlip(p=0.1),

        ShiftScaleRotate(shift_limit=1/14, scale_limit=0.1, rotate_limit=15, p=0.9),
        
        OpticalDistortion(distort_limit=0.1, shift_limit=0.05, p=0.5),
       
        # You can convert the augmented image back to its original
        # data type by using `FromFloat`.
        # FromFloat(dtype='uint16'),

        # As in `ToFloat` you can specify a `max_value` argument and all input values
        # will be multiplied by it.
        FromFloat(dtype='uint16', max_value=65535.0),

    ], p=p, additional_targets={"image1":"image"})


def augmentImagePairPhantom(image1, image2, size, output_path1, output_path2, augment):
    """
    Augment image pair
    """

    # resize
    image1R = cv2.resize(image1, (size,size), interpolation = cv2.INTER_LANCZOS4 )
    image2R = cv2.resize(image2, (size,size), interpolation = cv2.INTER_LANCZOS4 )
    if augment:
        # transform
        augmentation = light_aug(p=1)
        data = {"image": image1R, "image1": image2R}
        augmented = augmentation(**data)
        image1T, image2T = augmented["image"], augmented["image1"]
    else:
        image1T, image2T = [image1R , image2R]

    # Save
    cv2.imwrite(output_path1, image1T.astype(np.uint16) )
    cv2.imwrite(output_path2, image2T.astype(np.uint16) )


def main(args):
    image1 = cv2.imread(args.img1, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(args.img2, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)

    augmentation = light_aug(p=1)
    data = {"image": image1, "image1": image1}
    titles = ['Image 1', 'Image 2', 'Image 1 T', 'Image 2 T']


    for n in range(1,20):
        augmented = augmentation(**data)
        transformedImage, transformedImage2 = augmented["image"], augmented["image1"]
        images = [image1, image2, transformedImage, transformedImage2]

        plt.figure(figsize=(8, 8))
    
        for i in range(4):
            plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
            plt.title(titles[i])
            plt.xticks([]),plt.yticks([])
        plt.show()
        ret = cv2.imwrite(args.aimg1,transformedImage)
        ret = cv2.imwrite(args.aimg2,transformedImage)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BoneSuppression v2 - augmentation data')
    parser.add_argument('--img1', default='i_01.png', type=str, help='image 1')
    parser.add_argument('--img2', default='i_02.png', type=str, help='image 2')
    parser.add_argument('--aimg1', default='a_01.png', type=str, help='aug image 1')
    parser.add_argument('--aimg2', default='a_02.png', type=str, help='aug image 2')
    args = parser.parse_args()
    main(args)

