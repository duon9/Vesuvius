import torch
import numpy as np
from typing import Dict
from skimage.morphology import skeletonize, dilation
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform

class SkeletonTransform(BasicTransform):
    def __init__(self, do_tube: bool = True):
        """
        Tính toán skeleton cho từng slice dọc theo trục Z.
        Chỉ những pixel có giá trị bằng 1 mới được chuyển thành ảnh nhị phân để làm skeleton.
        """
        super().__init__()
        self.do_tube = do_tube

    def apply(self, data_dict: Dict, **params):
        """
        data_dict['segmentation'] shape: (1, Depth, Height, Width)
        """
        # Chuyển từ Tensor sang Numpy
        seg_all = data_dict["segmentation"].numpy()
        
        # Lấy khối dữ liệu kênh duy nhất (D, H, W)
        seg_3d = seg_all[0]
        
        # Khởi tạo mảng kết quả
        seg_all_skel = np.zeros_like(seg_3d, dtype=np.int16)

        # Duyệt qua từng slice theo trục Z
        for z in range(seg_3d.shape[0]):
            slice_2d = seg_3d[z]
            
            # Kiểm tra xem slice có chứa nhãn 1 không
            if np.any(slice_2d == 1):
                # 1. Chuyển thành ảnh nhị phân bằng cách lấy các pixel == 1
                bin_slice = (slice_2d == 1)
                
                # 2. Tạo skeleton 2D (độ dày 1 pixel)
                skel = skeletonize(bin_slice)
                
                # 3. Nếu do_tube=True, giãn nở đường skeleton
                if self.do_tube:
                    # Dilation 2 lần tạo ống độ dày ~5px
                    skel = dilation(dilation(skel))
                
                # 4. Gán kết quả vào khối 3D (giá trị trả về là 0 hoặc 1)
                # Vì ta chỉ làm cho nhãn 1, nên kết quả cuối cùng cũng mang giá trị 1
                seg_all_skel[z] = skel.astype(np.int16)

        # Thêm lại dimension kênh (1, D, H, W) và trả về Tensor
        data_dict["skel"] = torch.from_numpy(seg_all_skel[None])
        
        return data_dict