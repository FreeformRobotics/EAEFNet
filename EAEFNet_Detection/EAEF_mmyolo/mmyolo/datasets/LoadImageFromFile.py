from mmcv.transforms.builder import TRANSFORMS
import cv2
import numpy as np
from typing import Optional

import mmengine
import numpy as np

import mmcv
from mmcv.transforms.base import BaseTransform
from mmcv.transforms.builder import TRANSFORMS

@TRANSFORMS.register_module()
class LoadImage(BaseTransform):
    """Load an image from file.
    Required Keys:
    - img_path
    Modified Keys:
    - img
    - img_shape
    - ori_shape
    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:`mmcv.imfrombytes`.
            See :func:`mmcv.imfrombytes` for details.
            Defaults to 'cv2'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmengine.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        ignore_empty (bool): Whether to allow loading empty image or file path
            not existent. Defaults to False.
    """

    def __init__(self,
                 to_float32: bool = False,
                 color_type: str = 'color',
                 imdecode_backend: str = 'cv2',
                 file_client_args: dict = dict(backend='disk'),
                 ignore_empty: bool = False) -> None:
        self.ignore_empty = ignore_empty
        self.to_float32 = to_float32
        self.color_type = color_type
        self.imdecode_backend = imdecode_backend
        self.file_client_args = file_client_args.copy()
        self.file_client = mmengine.FileClient(**self.file_client_args)

    def transform(self, results: dict) -> Optional[dict]:
        """Functions to load image.
        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.
        Returns:
            dict: The dict contains loaded image and meta information.
        """

        filename = results['img_path']
        try:
            img = cv2.imread(results['img_path'],cv2.IMREAD_UNCHANGED)
        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e
        if self.to_float32:
            img = img.astype(np.float32)

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'ignore_empty={self.ignore_empty}, '
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f"imdecode_backend='{self.imdecode_backend}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str
