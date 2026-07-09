import torch
print("PyTorch 版本:", torch.__version__)
x = torch.rand(3, 3)
print(x)
print("FFT:", torch.fft.fft(x))

