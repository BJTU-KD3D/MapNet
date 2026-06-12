import torch
import numpy as np
from utils.space_filling import preprocess_pointcloud

def collate_fn_with_sfc(batch, method="circular"):
    """
    批处理函数，添加空间填充曲线排序
    
    参数:
    -------
    batch: 一批数据，每个元素包含点云坐标、特征、标签等
    method: 空间填充曲线方法，可选值：
        "standard" - 标准的Hilbert/Morton曲线
        "spherical" - 球坐标系编码 
        "cylindrical" - 柱坐标系编码
        "circular" - 圆形填充方式，针对圆形结构优化
        "auto" - 自动选择最合适的方法
    
    返回:
    -------
    批处理后的数据，点云已按空间填充曲线排序
    """
    coord, feat, target = list(zip(*batch))
    offset, count = [], 0
    
    # 应用空间填充曲线排序
    sorted_coord = []
    sorted_feat = []
    sorted_target = []
    
    for i, (c, f, t) in enumerate(zip(coord, feat, target)):
        # 检查输入类型并正确处理
        if isinstance(c, np.ndarray):
            c_tensor = torch.from_numpy(c).float()
        elif isinstance(c, torch.Tensor):
            c_tensor = c.float()
        else:
            raise TypeError(f"Unsupported coordinate type: {type(c)}")
            
        # 同样处理特征和标签
        if f is not None:
            if isinstance(f, np.ndarray):
                f_tensor = torch.from_numpy(f).float()
            elif isinstance(f, torch.Tensor):
                f_tensor = f.float()
            else:
                raise TypeError(f"Unsupported feature type: {type(f)}")
        else:
            f_tensor = None
            
        if t is not None:
            if isinstance(t, np.ndarray):
                t_tensor = torch.from_numpy(t).long()
            elif isinstance(t, torch.Tensor):
                t_tensor = t.long()
            else:
                raise TypeError(f"Unsupported target type: {type(t)}")
        else:
            t_tensor = None
        
        # 应用空间填充曲线排序，使用指定方法
        # 注意：对于圆形物体，推荐使用"circular"或"spherical"方法
        sorted_c, sorted_f, indices = preprocess_pointcloud(
            c_tensor, f_tensor, use_hilbert=True, method=method
        )
        
        # 同样重排标签
        sorted_t = t_tensor[indices] if t_tensor is not None else None
        
        # 添加到结果列表
        sorted_coord.append(sorted_c)
        sorted_feat.append(sorted_f if sorted_f is not None else torch.zeros_like(sorted_c))
        sorted_target.append(sorted_t)
        
        # 更新offset
        count += len(sorted_c)
        offset.append(count)
    
    # 合并所有数据
    sorted_coord_tensor = torch.cat(sorted_coord, dim=0)
    sorted_feat_tensor = torch.cat(sorted_feat, dim=0)
    sorted_target_tensor = torch.cat(sorted_target, dim=0)
    
    return sorted_coord_tensor, sorted_feat_tensor, sorted_target_tensor, torch.IntTensor(offset)

def collate_fn_with_multi_sfc(batch, methods=None):
    """
    批处理函数，使用多种优化的空间填充曲线方法
    
    参数:
    -------
    batch: 一批数据，每个元素包含点云坐标、特征、标签等
    methods: 要使用的空间填充曲线方法列表
    
    返回:
    -------
    批处理后的数据
    """
    if methods is None:
        # 使用优化的空间填充曲线方法组合 - 删除auto方法，避免错误
        methods = [
            "density",      # 处理点云密度不均匀情况
            "geometry",     # 利用几何结构信息
            "hierarchical", # 分层处理大型点云
            "circular"      # 适合室内圆形结构
        ]
        
    coord, feat, target = list(zip(*batch))
    offset, count = [], 0
    
    # 处理每个批次样本
    processed_coord, processed_feat, processed_target = [], [], []
    
    for i, (c, f, t) in enumerate(zip(coord, feat, target)):
        # 检查输入类型
        c_tensor = torch.from_numpy(c).float() if isinstance(c, np.ndarray) else c.float()
        f_tensor = torch.from_numpy(f).float() if isinstance(f, np.ndarray) else f.float() if f is not None else None
        t_tensor = torch.from_numpy(t).long() if isinstance(t, np.ndarray) else t.long() if t is not None else None
        
        # 每个批次使用不同方法，增加训练多样性
        # 使用固定的方法循环，不再使用auto方法
        selected_method = methods[i % len(methods)]
        
        try:
            # 应用选定的空间填充曲线排序
            sorted_c, sorted_f, indices = preprocess_pointcloud(
                c_tensor, f_tensor, use_hilbert=True, method=selected_method
            )
            
            # 排序标签
            sorted_t = t_tensor[indices] if t_tensor is not None else None
        except Exception as e:
            print(f"排序错误({selected_method}): {e}，使用原始顺序")
            sorted_c, sorted_f, sorted_t = c_tensor, f_tensor, t_tensor
            
        # 添加到处理后的列表
        processed_coord.append(sorted_c)
        processed_feat.append(sorted_f if sorted_f is not None else torch.zeros_like(sorted_c))
        processed_target.append(sorted_t)
        
        # 更新offset
        count += len(sorted_c)
        offset.append(count)
    
    # 合并处理后的数据
    processed_coord_tensor = torch.cat(processed_coord, dim=0)
    processed_feat_tensor = torch.cat(processed_feat, dim=0)
    processed_target_tensor = torch.cat(processed_target, dim=0)
    
    return processed_coord_tensor, processed_feat_tensor, processed_target_tensor, torch.IntTensor(offset)
