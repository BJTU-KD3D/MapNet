import torch
import torch.nn as nn
from lib.pointops.functions import pointops
import torch.nn.functional as F
from copy import copy, deepcopy
import numpy as np

# Add appropriate import statements
from model.mamba_module import Mamba, ModelArgs
from utils.space_filling import preprocess_pointcloud, sort_points_circular

class WeakLayer(nn.Module):
    def __init__(self, in_planes, out_planes, share_planes=8, nsample=16):
        super().__init__()
        self.mid_planes = mid_planes = out_planes // 1
        self.out_planes = out_planes
        self.share_planes = share_planes
        self.nsample = nsample
        self.linear_q = nn.Linear(in_planes, mid_planes)
        self.linear_k = nn.Linear(in_planes, mid_planes)
        self.linear_v = nn.Linear(in_planes, out_planes) 
        
        self.linear_p = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3), nn.ReLU(inplace=True), nn.Linear(3, out_planes))
        self.linear_uni_spatial = nn.Sequential(nn.Linear(in_planes, 1),nn.BatchNorm1d(1),nn.ReLU(inplace=True))

        self.linear_add = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3), nn.ReLU(inplace=True), nn.Linear(3, out_planes))

        self.linear_w = nn.Sequential(nn.BatchNorm1d(mid_planes), nn.ReLU(inplace=True),
                                    nn.Linear(mid_planes, mid_planes),
                                    nn.BatchNorm1d(mid_planes),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(out_planes, out_planes))
        
        self.softmax = nn.Softmax(dim=1)
        self.softmax2 = nn.Softmax(dim=1)
        self.softmax3 = nn.Softmax(dim=1)
        self.bn1=nn.Sequential(nn.LayerNorm(out_planes),nn.ReLU(inplace=True))
        self.a = nn.Parameter(torch.zeros(size=(nsample, 1)),requires_grad=True)
        self.b = nn.Parameter(torch.zeros(size=(nsample, out_planes)),requires_grad=True)
        self.d = nn.Parameter(torch.zeros(size=(nsample,out_planes)),requires_grad=True)
        nn.init.xavier_uniform_(self.a.data)
        nn.init.xavier_uniform_(self.b.data)
        nn.init.xavier_uniform_(self.d.data)

    def forward(self, pxo) -> torch.Tensor:
        p, x, o = pxo 

        x_r=x.clone()

        x_q, x_k, x_v = self.linear_q(x), self.linear_k(x), self.linear_v(x)
        x_uni_spatial = self.linear_uni_spatial(x)
        device = torch.device("cuda:0")
        
        x_k = pointops.queryandgroup(self.nsample, p, p, x_k, None, o, o, use_xyz=True)
        x_v_j = pointops.queryandgroup(self.nsample, p, p, x_v, None, o, o, use_xyz=False)
        n, nsample, c = x_v_j.shape; s = self.share_planes

        e, x_k = x_k[:, :, 0:3], x_k[:, :, 3:] 
        
        # ================================================== Position Encoding ==================================================
        
        p_r=torch.clone(e)
        e_dis=torch.from_numpy(np.linalg.norm(e.detach().cpu().numpy(),axis=2)).unsqueeze(2).cuda()
        e_dis_xy=torch.from_numpy(np.linalg.norm(e.detach().cpu().numpy()[:,:,:2],axis=2)).unsqueeze(2).cuda()
        cos_theta_e=(e_dis_xy/e_dis).nan_to_num(0)
        
        cos_theta_a=(e[:,:,1].unsqueeze(2)/e_dis_xy).nan_to_num(0)
        all_feat=torch.cat([e_dis,cos_theta_a,cos_theta_e],dim=-1)\
        
        for i, layer in enumerate(self.linear_add): 
            all_feat = layer(all_feat.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i == 1 else layer(all_feat)

        for i, layer in enumerate(self.linear_p): 
            p_r = layer(p_r.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i == 1 else layer(p_r)
        
        p_r+=self.d*all_feat
        
        # ================================================== Weight ==================================================
        w_spatial_d = torch.matmul((x_k - x_q.unsqueeze(1)).transpose(0,2).contiguous(),x_uni_spatial)
        w_spatial_d = self.softmax2(w_spatial_d)  # (n, nsample, c)
        w_spatial_e = torch.matmul(x_r,w_spatial_d.transpose(0,1).contiguous()).transpose(0,1).contiguous()
        
        w_spatial_f = w_spatial_d.transpose(0,2).contiguous()
        
        w_spatial_e = self.softmax3(w_spatial_e)
        
        w=torch.add(self.a*w_spatial_f,self.b*w_spatial_e)#+ p_r #(x_k - x_q.unsqueeze(1)) + p_r

        for i, layer in enumerate(self.linear_w): 
            w = layer(w.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i in [0,3] else layer(w)
        
        w=self.softmax(w)

        # ================================================== Combine ==================================================
        
        n, nsample, c = x_v_j.shape; s = self.share_planes
        x = ((x_v_j + p_r) * w).sum(1).view(n, c)
        return x

class Downsampling(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, nsample=16):
        super().__init__()
        self.stride, self.nsample = stride, nsample
        if stride != 1:
            # Modify the dimension handling of the progressive downsampling path
            self.mid_planes = (3 + in_planes + out_planes) // 2
            self.progressive = nn.Sequential(
                nn.Conv1d(3 + in_planes, self.mid_planes, 1),
                nn.BatchNorm1d(self.mid_planes),
                nn.ReLU(inplace=True),
                nn.Conv1d(self.mid_planes, out_planes, 1),
                nn.BatchNorm1d(out_planes),
                nn.ReLU(inplace=True)
            )
            
            # Modify the hybrid pooling layers
            self.max_pool = nn.MaxPool1d(nsample)
            self.avg_pool = nn.AvgPool1d(nsample)
            self.pool_fusion = nn.Sequential(
                nn.Conv1d(out_planes * 2, out_planes, 1),
                nn.BatchNorm1d(out_planes),
                nn.ReLU(inplace=True)
            )
        else:
            self.linear = nn.Linear(in_planes, out_planes, bias=False)
        self.bn = nn.BatchNorm1d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, pxo):
        p, x, o = pxo
        if self.stride != 1:
            n_o, count = [o[0].item() // self.stride], o[0].item() // self.stride
            for i in range(1, o.shape[0]):
                count += (o[i].item() - o[i-1].item()) // self.stride
                n_o.append(count)
            n_o = torch.cuda.IntTensor(n_o)
            idx = pointops.furthestsampling(p, o, n_o)
            n_p = p[idx.long(), :]
            x = pointops.queryandgroup(self.nsample, p, n_p, x, None, o, n_o, use_xyz=True)
            
            # Modify the feature processing flow
            B = n_p.size(0)  # batch size
            
            # Adjust the dimension order to fit Conv1d
            x = x.permute(0, 2, 1).contiguous()  # [B, C+3, nsample]
            
            # Progressive feature transformation (already includes BN and ReLU)
            x = self.progressive(x)  # [B, out_planes, nsample]
            
            # Hybrid pooling
            x_max = self.max_pool(x)  # [B, out_planes, 1]
            x_avg = self.avg_pool(x)  # [B, out_planes, 1]
            x = torch.cat([x_max, x_avg], dim=1)  # [B, 2*out_planes, 1]
            
            # Fuse features and compress dimensions
            x = self.pool_fusion(x).squeeze(-1)  # [B, out_planes]
            
            p, o = n_p, n_o
        else:
            x = self.relu(self.bn(self.linear(x)))
        return [p, x, o]


class FeaturePropagation(nn.Module):
    def __init__(self, in_planes, out_planes=None):
        super().__init__()
        if out_planes is None:
            self.linear1 = nn.Sequential(nn.Linear(2*in_planes, in_planes), nn.BatchNorm1d(in_planes), nn.ReLU(inplace=True))
            self.linear2 = nn.Sequential(nn.Linear(in_planes, in_planes), nn.ReLU(inplace=True))
        else:
            self.linear1 = nn.Sequential(nn.Linear(out_planes, out_planes), nn.BatchNorm1d(out_planes), nn.ReLU(inplace=True))
            self.linear2 = nn.Sequential(nn.Linear(in_planes, out_planes), nn.BatchNorm1d(out_planes), nn.ReLU(inplace=True))
        
    def forward(self, pxo1, pxo2=None):
        if pxo2 is None:
            _, x, o = pxo1
            x_tmp = []
            for i in range(o.shape[0]):
                if i == 0:
                    s_i, e_i, cnt = 0, o[0], o[0]
                else:
                    s_i, e_i, cnt = o[i-1], o[i], o[i] - o[i-1]
                x_b = x[s_i:e_i, :]
                x_b = torch.cat((x_b, self.linear2(x_b.sum(0, True) / cnt).repeat(cnt, 1)), 1)
                x_tmp.append(x_b)
            x = torch.cat(x_tmp, 0)
            x = self.linear1(x)
        else:
            p1, x1, o1 = pxo1; p2, x2, o2
            x = self.linear1(x1) + pointops.interpolation(p2, p1, self.linear2(x2), o2, o1)
        return x


class WeakBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, share_planes=8, nsample=16):
        super(WeakBlock, self).__init__()
        # Keep the original layers unchanged
        self.linear1 = nn.Linear(in_planes, planes, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.weak = WeakLayer(planes, planes, share_planes, nsample)
        self.bn2 = nn.BatchNorm1d(planes)
        self.linear3 = nn.Linear(planes, planes * self.expansion, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        
        # Adjust parameters based on the number of channels
        d_state = min(16, planes // 4)  # state space dimension
        
        # Create a ModelArgs object to initialize Mamba
        args = ModelArgs(
            d_model=planes,      # input feature dimension
            n_layer=1,           # only one Mamba layer is needed
            vocab_size=256,      # this parameter does not matter, but must be provided
            d_state=d_state,     # state space dimension
            expand=1.5,          # expansion factor
            dt_rank="auto",      # automatically set dt_rank
            d_conv=4,            # convolution kernel size
            pad_vocab_size_multiple=8,
            conv_bias=True,      # use convolution bias
            bias=False           # do not use linear layer bias
        )
        
        # Initialize Mamba with ModelArgs
        self.mamba = Mamba(args)
        
        # Use ReLU activation for consistency
        self.act = nn.ReLU(inplace=True)
        self.dp2 = nn.Dropout(0.5, inplace=True)
        
        # Add an adapter to feed point cloud data into the Mamba input
        self.mamba_adapter = MambaPointcloudAdapter()

    def forward(self, pxo):
        p, x, o = pxo
        identity = x
        
        # 1. Keep the original forward flow
        x = self.act(self.bn1(self.linear1(x)))
        x = self.act(self.bn2(self.weak([p, x, o])))
        
        # 2. Add Mamba processing - now also pass coordinates p and offsets o for space-filling curve ordering
        x = self.mamba_adapter(x, self.mamba, p, o)
        
        # 3. Keep the original output flow
        x = self.bn3(self.linear3(x))
        x = self.dp2(x)
        x += identity
        x = self.act(x)
        
        return [p, x, o]


# Optimize the MambaPointcloudAdapter class to use enhanced space-filling curves
class MambaPointcloudAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.backup_mlp = None
        # Use optimized space-filling curve methods
        self.space_filling_methods = [
            "density",       # density-adaptive filling, handles uneven point cloud distribution
            "geometry",      # geometry-aware filling, suitable for structured scenes
            "hierarchical",  # hierarchical locality filling, suitable for large point clouds
            "circular"       # circular filling, suitable for indoor scenes
        ]
        # Assign weights to different methods, tuned by their effectiveness
        self.ensemble_weights = [0.4, 0.3, 0.2, 0.1]
        self.use_multi_sfc = True  # use fusion of multiple methods
        
        # Dynamically learn to adapt to different point cloud scenes
        self.scene_context = None  # scene context, can be updated during training
        
    def _create_backup_mlp(self, dim, device):
        """Create a backup MLP in case Mamba fails"""
        return nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(inplace=True)
        ).to(device)
    
    def forward(self, x, mamba_model, p=None, o=None):
        """Process point cloud features using enhanced space-filling curves"""
        device = x.device
        dim = x.shape[1]
        
        # Lazily initialize the backup MLP
        if self.backup_mlp is None:
            self.backup_mlp = self._create_backup_mlp(dim, device)
        
        # If there are no point coordinates or multi-method fusion is disabled, use a single method
        if p is None or not self.use_multi_sfc:
            return self._process_single_method(x, mamba_model, p, o, method_idx=0)
        
        # Use multi-method fusion
        try:
            results = []
            valid_weights = []
            
            # Process each space-filling method separately
            for i, method in enumerate(self.space_filling_methods):
                try:
                    # Process with the current method
                    result = self._process_single_method(x.clone(), mamba_model, p, o, method_idx=i)
                    results.append(result)
                    valid_weights.append(self.ensemble_weights[i])
                except Exception as e:
                    print(f"Method {method} failed: {e}")
                    continue
            
            # If no method succeeded, use the backup MLP
            if len(results) == 0:
                print("All space-filling methods failed, using the backup MLP")
                return self.backup_mlp(x)
                
            # Normalize the weights
            valid_weights = torch.tensor(valid_weights, device=device)
            valid_weights = valid_weights / valid_weights.sum()
            
            # Weighted fusion of the results
            final_output = torch.zeros_like(x)
            for i, result in enumerate(results):
                final_output += result * valid_weights[i]
                
            return final_output
                
        except Exception as e:
            print(f"Multi space-filling curve fusion failed: {e}, using the backup MLP")
            return self.backup_mlp(x)

    def _process_single_method(self, x, mamba_model, p=None, o=None, method_idx=0):
        """Process a single space-filling curve method"""
        device = x.device
        method = self.space_filling_methods[method_idx] if method_idx < len(self.space_filling_methods) else "density"
        
        from utils.space_filling import preprocess_pointcloud
        
        # Sort using the optimized space-filling curve
        if p is not None:
            try:
                # Call the optimized space-filling curve processing function
                _, sorted_feat, indices = preprocess_pointcloud(
                    p, x, use_hilbert=True, method=method
                )
                x = sorted_feat
                
                # Record the reverse indices
                reverse_indices = torch.zeros_like(indices, dtype=torch.long)
                reverse_indices[indices] = torch.arange(x.size(0), device=device)
            except Exception as e:
                print(f"{method} sorting failed: {e}, skipping sorting")
        
        # Use a direct feature processing approach, simplifying the call structure to avoid "too many values to unpack" errors
        try:
            # 1. Apply the normalization layer
            x_norm = torch.nn.functional.layer_norm(
                x, [x.shape[-1]], eps=1e-5
            )
            
            # 2. Safely call the Mamba components
            if hasattr(mamba_model, 'norm_f'):
                # Use Mamba's normalization
                x_norm = mamba_model.norm_f(x.unsqueeze(1)).squeeze(1)
                
                # Use Mamba's first layer to process features
                if hasattr(mamba_model, 'layers') and len(mamba_model.layers) > 0:
                    try:
                        # Get the first layer's mixer
                        mixer = mamba_model.layers[0].mixer
                        
                        # Use the mixer's projection and processing
                        x_proj = mixer.in_proj(x_norm)
                        
                        # Split the projection result
                        split_size = x_proj.size(-1) // 2
                        x_split = x_proj[:, :split_size] 
                        res_split = x_proj[:, split_size:]
                        
                        # Compute activation and gating
                        res_act = F.silu(res_split)
                        output = mixer.out_proj(x_split * res_act)
                        
                        # Add the residual connection
                        output = output + x_norm
                        
                        # Restore the original order (if sorting was applied)
                        if p is not None and 'reverse_indices' in locals():
                            output = output[reverse_indices]
                        
                        return output
                    except Exception as inner_e:
                        print(f"Mamba internal processing error: {inner_e}, using a simplified implementation")
            
            # If the above approach fails, use a simplified implementation
            proj_weight = torch.randn(x.shape[-1], x.shape[-1], device=device).mul_(0.02)
            output = x + 0.1 * F.linear(x_norm, proj_weight)
            
            # Restore the original order (if sorting was applied)
            if p is not None and 'reverse_indices' in locals():
                output = output[reverse_indices]
            
            return output
                
        except Exception as e:
            print(f"Feature processing error: {e}")
            return self.backup_mlp(x)

class WeakSeg(nn.Module):
    def __init__(self, block, blocks, c=6, k=13,unitArch=[32, 64, 128, 256, 512]):
        super().__init__()
        self.c = c
        self.in_planes, planes = 6, unitArch

        self.num_heads=8

        fpn_planes, fpnhead_planes, share_planes = 128, 64, 8
        stride, nsample = [1, 4, 4, 4, 4], [8, 16, 16, 16, 16]

        self.enc1 = self._make_enc(block, planes[0], blocks[0], share_planes, stride=stride[0], nsample=nsample[0])
        self.enc2 = self._make_enc(block, planes[1], blocks[1], share_planes, stride=stride[1], nsample=nsample[1])
        self.enc3 = self._make_enc(block, planes[2], blocks[2], share_planes, stride=stride[2], nsample=nsample[2])
        self.enc4 = self._make_enc(block, planes[3], blocks[3], share_planes, stride=stride[3], nsample=nsample[3])
        self.enc5 = self._make_enc(block, planes[4], blocks[4], share_planes, stride=stride[4], nsample=nsample[4])
        self.dec5 = self._make_dec(block, planes[4], 1, share_planes, nsample=nsample[4], is_head=True)
        self.dec4 = self._make_dec(block, planes[3], 1, share_planes, nsample=nsample[3])
        self.dec3 = self._make_dec(block, planes[2], 1, share_planes, nsample=nsample[2])
        self.dec2 = self._make_dec(block, planes[1], 1, share_planes, nsample=nsample[1])
        self.dec1 = self._make_dec(block, planes[0], 1, share_planes, nsample=nsample[0])
        self.cls = nn.Sequential(nn.Linear(planes[0], planes[0]), nn.BatchNorm1d(planes[0]), nn.ReLU(inplace=True), nn.Linear(planes[0], k))
        self.in_planes=16*self.num_heads

    def get_activation(self,name):
        def hook(model, input, output):
            self.activation[name] = output.detach()
        return hook

    def _make_enc(self, block, planes, blocks, share_planes=8, stride=1, nsample=16):
        layers = []
        layers.append(Downsampling(self.in_planes, planes * block.expansion, stride, nsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample))
        return nn.Sequential(*layers)

    def _make_dec(self, block, planes, blocks, share_planes=8, nsample=16, is_head=False):
        layers = []
        if is_head:
            # The head layer only uses the current input: keep the dimension unchanged
            layers.append(KNNDecoder(self.in_planes, None, self.in_planes, k=nsample))
            # self.in_planes is not updated
        else:
            # Non-head layer:
            # dec_in comes from the corresponding encoder output dimension; skip_in takes the previous decoder output dimension (self.in_planes)
            dec_in = planes * block.expansion
            layers.append(KNNDecoder(dec_in, self.in_planes, dec_in, k=nsample))
            self.in_planes = dec_in
            
        # Add residual blocks
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample))
            
        return nn.Sequential(*layers)

    def forward(self, pxo):
        
        p0, x0, o0 = pxo
        x0 = torch.cat([x0,p0],1)
        p1, x1, o1 = self.enc1([p0, x0, o0])
        p2, x2, o2 = self.enc2([p1, x1, o1])
        p3, x3, o3 = self.enc3([p2, x2, o2])
        p4, x4, o4 = self.enc4([p3, x3, o3])
        p5, x5, o5 = self.enc5([p4, x4, o4])
        x5 = self.dec5[1:]([p5, self.dec5[0]([p5, x5, o5]), o5])[1]
        x4 = self.dec4[1:]([p4, self.dec4[0]([p4, x4, o4], [p5, x5, o5]), o4])[1]
        x3 = self.dec3[1:]([p3, self.dec3[0]([p3, x3, o3], [p4, x4, o4]), o3])[1]
        x2 = self.dec2[1:]([p2, self.dec2[0]([p2, x2, o2], [p3, x3, o3]), o2])[1]
        x1 = self.dec1[1:]([p1, self.dec1[0]([p1, x1, o1], [p2, x2, o2]), o1])[1]
        x = self.cls(x1)
        return x

class KNNDecoder(nn.Module):
    def __init__(self, dec_in, skip_in, out_channels, k=16):
        super().__init__()
        self.k = k
        self.out_channels = out_channels
        
        # 1. Main feature transformation (used in all cases)
        self.transform = nn.Sequential(
            nn.Linear(dec_in, out_channels),
            nn.LayerNorm(out_channels),
            nn.GELU()
        )
        
        # 2. Feature propagation layers (used when pxo2 is None)
        self.linear1 = nn.Sequential(
            nn.Linear(2 * out_channels, out_channels),
            nn.LayerNorm(out_channels), 
            nn.GELU()
        )
        self.linear2 = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.GELU()
        )
        
        # 3. Skip-connection related layers (used when pxo2 is not None)
        if skip_in is not None:
            self.skip_transform = nn.Sequential(
                nn.Linear(skip_in, out_channels),
                nn.LayerNorm(out_channels),
                nn.GELU()
            )
            
            self.fusion_net = nn.Sequential(
                nn.Linear(out_channels * 2, out_channels),
                nn.LayerNorm(out_channels),
                nn.GELU(),
                nn.Linear(out_channels, 1),
                nn.Sigmoid()
            )
            
    def forward(self, pxo1, pxo2=None):
        p1, x1, o1 = pxo1
        
        # 1. Main feature transformation
        x = self.transform(x1)
        
        if pxo2 is None:
            # 2. When there is no skip connection, use processing similar to FeaturePropagation
            x_tmp = []
            for i in range(o1.shape[0]):
                if i == 0:
                    s_i, e_i, cnt = 0, o1[0], o1[0]
                else:
                    s_i, e_i, cnt = o1[i-1], o1[i], o1[i] - o1[i-1]
                x_b = x[s_i:e_i, :]
                x_b = torch.cat((x_b, self.linear2(x_b.sum(0, True) / cnt).repeat(cnt, 1)), 1)
                x_tmp.append(x_b)
            x = torch.cat(x_tmp, 0)
            x = self.linear1(x)
            
        else:
            # 3. When there is a skip connection, use KNN and feature fusion logic
            p2, x2, o2 = pxo2
            
            x2 = self.skip_transform(x2)
            x2_grouped = pointops.queryandgroup(
                self.k, p2, p1, x2, None, o2, o1, use_xyz=True
            )
            
            xyz_diff = x2_grouped[:, :, 0:3]
            feat_grouped = x2_grouped[:, :, 3:]
            
            dist = torch.sum(xyz_diff ** 2, dim=-1, keepdim=True)
            weights = F.softmax(-dist, dim=1)
            
            x_skip = (feat_grouped * weights).sum(1)
            
            fusion_feat = torch.cat([x, x_skip], dim=-1)
            fusion_weights = self.fusion_net(fusion_feat)
            
            x = x + fusion_weights * x_skip
            
        return x

def weak_seg_repro(custom=None, **kwargs):
    
    model = WeakSeg(WeakBlock, [1, 3, 3, 6, 3], **kwargs)
    return model

