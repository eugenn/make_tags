from typing import Tuple

import numpy as np
import cv2
import math
import albumentations as A
from albumentations.core.transforms_interface import DualTransform


class _3DRotateTransform(DualTransform):
    def __init__(self, theta, phi, gamma, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.theta = theta
        self.phi = phi
        self.gamma = gamma
        self.dx = 0
        self.dy = 0
        self.dz = 0

    def apply(self, image, **params):
        height = image.shape[0]
        width = image.shape[1]
        num_channels = image.shape[2]

        t = math.radians(self.gamma)
        rot_side = width * (math.cos(t) + math.sin(t))
        safe_border = int(rot_side - width)
        image = cv2.copyMakeBorder(image, safe_border, safe_border, safe_border, safe_border,
                                   cv2.BORDER_CONSTANT, value=[0, 0, 0])
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)

        # Get radius of rotation along 3 axes
        rtheta, rphi, rgamma = self.get_rad(self.theta, self.phi, self.gamma)

        # Get ideal focal length on z axis
        # NOTE: Change this section to other axis if needed
        d = np.sqrt(height ** 2 + width ** 2)
        self.focal = d / (2 * np.sin(rgamma) if np.sin(rgamma) != 0 else 1)

        # Get projection matrix
        mat = self.get_M(rtheta, rphi, rgamma, self.dx, self.dy, self.focal, width, height)

        return cv2.warpPerspective(image.copy(), mat, (width, height), borderMode=0)

    def apply_to_mask(self, image, **params):
        return self.apply(image, **params)

    def get_M(self, theta, phi, gamma, dx, dy, dz, w, h):
        # w = self.width
        # h = self.height
        f = dz

        # Projection 2D -> 3D matrix
        A1 = np.array([[1, 0, -w / 2],
                       [0, 1, -h / 2],
                       [0, 0, 1],
                       [0, 0, 1]])

        # Rotation matrices around the X, Y, and Z axis
        RX = np.array([[1, 0, 0, 0],
                       [0, np.cos(theta), -np.sin(theta), 0],
                       [0, np.sin(theta), np.cos(theta), 0],
                       [0, 0, 0, 1]])

        RY = np.array([[np.cos(phi), 0, -np.sin(phi), 0],
                       [0, 1, 0, 0],
                       [np.sin(phi), 0, np.cos(phi), 0],
                       [0, 0, 0, 1]])

        RZ = np.array([[np.cos(gamma), -np.sin(gamma), 0, 0],
                       [np.sin(gamma), np.cos(gamma), 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])

        # Composed rotation matrix with (RX, RY, RZ)
        R = np.dot(np.dot(RX, RY), RZ)

        # Translation matrix
        T = np.array([[1, 0, 0, dx],
                      [0, 1, 0, dy],
                      [0, 0, 1, dz],
                      [0, 0, 0, 1]])

        # Projection 3D -> 2D matrix
        A2 = np.array([[f, 0, w / 2, 0],
                       [0, f, h / 2, 0],
                       [0, 0, 1, 0]])

        # Final transformation matrix
        return np.dot(A2, np.dot(T, np.dot(R, A1)))

    def get_rad(self, theta, phi, gamma):
        return (self.to_rad(theta),
                self.to_rad(phi),
                self.to_rad(gamma))

    def get_deg(self, rtheta, rphi, rgamma):
        return (self.to_deg(rtheta),
                self.to_deg(rphi),
                self.to_deg(rgamma))

    def to_rad(self, deg):
        return deg * np.pi / 180.0

    def to_deg(self, rad):
        return rad * 180.0 / np.pi

    def get_transform_init_args_names(self) -> Tuple[str, str, str]:
        return ("theta", "phi", "gamma")


def transform(kind, resize_scale=100, rot_min=-10, rot_max=45,
              defocus_rad=(3, 5), brightness_limit=[-0.2, 0.1], contrast_limit=0.2, blur_limit=7, prob=1):
    if kind == 'resize':
        aug_transform = A.Compose([
            A.Resize(height=resize_scale,
                     width=resize_scale,
                     interpolation=cv2.INTER_LANCZOS4,
                     p=prob),
        ])

    elif kind == 'rotate':
        aug_transform = A.Compose([
            A.Resize(height=100,
                     width=100,
                     interpolation=cv2.INTER_LANCZOS4,
                     p=1),
            A.SafeRotate(limit=[rot_min, rot_max],
                         interpolation=cv2.INTER_CUBIC,
                         border_mode=cv2.BORDER_CONSTANT,
                         p=prob),
            # A.Affine(rotate=0,
            #          shear=[-10, 20],
            #          interpolation=cv2.INTER_CUBIC,
            #          mode=cv2.BORDER_CONSTANT,
            #          keep_ratio=True,
            #          p=1)
        ])

    elif kind == 'defocus':
        aug_transform = A.Compose([
            A.Resize(height=100,
                     width=100,
                     interpolation=cv2.INTER_LANCZOS4,
                     p=1),

            A.Defocus(radius=defocus_rad, p=prob),
            # A.RandomBrightnessContrast(p=1),
        ])

    elif kind == 'motion_blur':
        aug_transform = A.Compose([
            A.Resize(height=100,
                     width=100,
                     interpolation=cv2.INTER_LANCZOS4,
                     p=1),

            A.MotionBlur(blur_limit=blur_limit, always_apply=True, p=0.5),
            # A.RandomBrightnessContrast(p=1),
        ])

    elif kind == 'contrast':
        aug_transform = A.Compose([
            A.Resize(height=100,
                     width=100,
                     interpolation=cv2.INTER_LANCZOS4,
                     p=1),

            # A.Defocus(radius=(3, 5), p=1),
            A.RandomBrightnessContrast(brightness_limit=brightness_limit, contrast_limit=contrast_limit, p=prob),
        ])
    elif kind == '3drotate':
        aug_transform = A.Compose([
            A.Resize(height=100,
                     width=100,
                     interpolation=cv2.INTER_LANCZOS4,
                     p=1),

            _3DRotateTransform(theta=10, phi=30, gamma=10, p=1.0),
        ])
    else:
        print("Invalid enty")
        return 0

    return aug_transform
