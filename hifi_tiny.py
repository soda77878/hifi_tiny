from tinygrad import Tensor, nn
from tinygrad.nn import Conv1d
from tinygrad import Device
from tinygrad.engine.jit import TinyJit

from tinygrad.nn.state import torch_load

def get_padding(kernel_size, dilation=1): return int((kernel_size*dilation - dilation)/2)

#LRELU_SLOPE = 0.1

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
    self.num_kernels, self.num_upsamples = len(resblock_kernel_sizes), len(upsample_rates)
    self.conv_pre = nn.Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
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


if __name__ == "__main__":
    #values from hifigan/config.py 
    initial_channel = 512 # upsample_initial_channel
    resblock = "1"
    resblock_kernel_sizes = [3, 7, 11]
    resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    upsample_rates = [8, 8, 2, 2]
    upsample_initial_channel = 512
    upsample_kernel_sizes = [16, 16, 4, 4]

    hifigan = Generator(initial_channel, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes) # HiFiGAN
    print("Generator object")

    import numpy as np
    mel = Tensor(np.load("mel_spectogram_test.npy")) #torch.Size([1, 80, 252])
    print("Mel spectogram prepared.:", mel.shape)

    out = hifigan.forward(mel) 
    print("Done")

    # PASS IT TO GENERATOR, NO WEIGHTS LOADED YET..