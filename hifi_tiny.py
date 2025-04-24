from tinygrad import Tensor, nn
from tinygrad.nn import Conv1d
from tinygrad import Device
from tinygrad.engine.jit import TinyJit

from tinygrad.nn.state import torch_load
import numpy as np
def get_padding(kernel_size, dilation=1): return int((kernel_size*dilation - dilation)/2)

LRELU_SLOPE = 0.1

class ResBlock1:
  def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
    self.convs1 = [nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[i], padding=get_padding(kernel_size, dilation[i])) for i in range(3)]
    self.convs2 = [nn.Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1)) for _ in range(3)]
  def forward(self, x: Tensor, x_mask=None):
    for c1, c2 in zip(self.convs1, self.convs2):
      xt = x.leaky_relu(LRELU_SLOPE)
      xt = c1(xt if x_mask is None else xt * x_mask).leaky_relu(LRELU_SLOPE)
      x = c2(xt if x_mask is None else xt * x_mask) + x
    return x if x_mask is None else x * x_mask


class ResBlock2:
  def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
    self.convs = [nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[i], padding=get_padding(kernel_size, dilation[i])) for i in range(2)]
  def forward(self, x, x_mask=None):
    for c in self.convs:
      xt = x.leaky_relu(LRELU_SLOPE)
      xt = c(xt if x_mask is None else xt * x_mask)
      x = xt + x
    return x if x_mask is None else x * x_mask

class Generator:
  def __init__(self, initial_channel, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=0):
    # initial_channel unused
    self.num_kernels, self.num_upsamples = len(resblock_kernel_sizes), len(upsample_rates)
    self.conv_pre = nn.Conv1d(80, upsample_initial_channel, 7, 1, padding=3)                           # Made a change here
    resblock = ResBlock1 if resblock == '1' else ResBlock2
    self.ups = [nn.ConvTranspose1d(upsample_initial_channel//(2**i), upsample_initial_channel//(2**(i+1)), k, u, padding=(k-u)//2) for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes))]
    self.resblocks = []
    self.upsample_rates = upsample_rates
    for i in range(len(self.ups)):
      ch = upsample_initial_channel // (2 ** (i + 1))
      for _, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
        self.resblocks.append(resblock(ch, k, d))
    self.conv_post = nn.Conv1d(ch, 1, 7, 1, padding=3, bias=False)
    if gin_channels != 0: self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)
  @TinyJit
  def forward(self, x: Tensor, g=None):
    x = self.conv_pre(x)
    if g is not None:  x = x + self.cond(g)
    for i in range(self.num_upsamples):
      x = self.ups[i](x.leaky_relu(LRELU_SLOPE))
      xs = sum(self.resblocks[i * self.num_kernels + j].forward(x) for j in range(self.num_kernels))
      x = (xs / self.num_kernels).realize()
    res = self.conv_post(x.leaky_relu()).tanh().realize()
    return res

class LayerNorm(nn.LayerNorm):
  def __init__(self, channels, eps=1e-5): super().__init__(channels, eps, elementwise_affine=True)
  def forward(self, x: Tensor): return self.__call__(x.transpose(1, -1)).transpose(1, -1)

def norm_except_dim(v, dim):
  if dim == -1: return np.linalg.norm(v)
  if dim == 0:
    (output_shape := [1] * v.ndim)[0] = v.shape[0]
    return np.linalg.norm(v.reshape(v.shape[0], -1), axis=1).reshape(output_shape)
  if dim == v.ndim - 1:
    (output_shape := [1] * v.ndim)[-1] = v.shape[-1]
    return np.linalg.norm(v.reshape(-1, v.shape[-1]), axis=0).reshape(output_shape)
  transposed_v = np.transpose(v, (dim,) + tuple(i for i in range(v.ndim) if i != dim))
  return np.transpose(norm_except_dim(transposed_v, 0), (dim,) + tuple(i for i in range(v.ndim) if i != dim))
def weight_norm(v: Tensor, g: Tensor, dim):
  v, g = v.numpy(), g.numpy()
  return Tensor(v * (g / norm_except_dim(v, dim)))
from tinygrad.nn.state import torch_load
import time
from pathlib import Path
def load_checkpoint(checkpoint_path, model):
  assert Path(checkpoint_path).is_file()
  start_time = time.time()
  checkpoint_dict = torch_load(checkpoint_path)

  #key_list = list(checkpoint_dict['generator'].keys())
  #print(key_list)
  
  saved_state_dict = checkpoint_dict['generator']
  weight_g, weight_v, parent = None, None, None
  for key, v in saved_state_dict.items():
    try:
      obj, skip = model, False
      for k in key.split('.'):
        if k.isnumeric(): obj = obj[int(k)]
        elif isinstance(obj, dict): obj = obj[k]
        else:
          if isinstance(obj, (LayerNorm, nn.LayerNorm)) and k in ["gamma", "beta"]:
            k = "weight" if k == "gamma" else "bias"
          elif k in ["weight_g", "weight_v"]:
            parent, skip = obj, True
            if k == "weight_g": weight_g = v
            else: weight_v = v
          if not skip: obj = getattr(obj, k)
      if weight_g is not None and weight_v is not None:
        setattr(obj, "weight_g", weight_g.numpy())
        setattr(obj, "weight_v", weight_v.numpy())
        obj, v = getattr(parent, "weight"), weight_norm(weight_v, weight_g, 0)
        weight_g, weight_v, parent, skip = None, None, None, False
        print('obj shape == v.shape',obj.shape == v.shape )
        obj.assign(v.to(obj.device))
      # if not skip and obj.shape == v.shape: obj.assign(v.to(obj.device))  
      # elif not skip: print(f"MISMATCH SHAPE IN {key}, {obj.shape} {v.shape}")
    except Exception as e: raise e
  print(f"Loaded checkpoint '{checkpoint_path}' (iteration .. in {time.time() - start_time:.4f}s")
  return model

if __name__ == "__main__":
    #values from hifigan/config.py 
    initial_channel = 512 # upsample_initial_channel
    resblock = "1"
    resblock_kernel_sizes = [3, 7, 11]
    resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    upsample_rates = [8, 8, 2, 2]
    upsample_initial_channel = 512
    upsample_kernel_sizes = [16, 16, 4, 4]
    HIFIGAN_CHECKPOINT = "model_ckpt/hifigan_T2_v1"

    hifigan = Generator(initial_channel, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes) # HiFiGAN
    _ = load_checkpoint(HIFIGAN_CHECKPOINT,hifigan)

    print("done loading weights..")

    
    mel = Tensor(np.load("mel_spectogram_test.npy")) #torch.Size([1, 80, 252])
    print("Mel spectogram prepared.:", mel.shape)

    out = hifigan.forward(mel) 

    out = out.float().squeeze().numpy()
    print('out shape:', out.shape)
    import soundfile as sf
    sf.write('jazzy.wav', out, 22050, 'PCM_24')

    # Checkpoint loading works... But output is grainyand low quality?? ..
    # Debug:
    # 1.) Problem with Loading Weights?       # this (?) 
    # 2.) Problem with Architecture ? (maybe..) 

    # Errro could possibly be related to not having init_weights()? 4/23/25
    print("Done")
