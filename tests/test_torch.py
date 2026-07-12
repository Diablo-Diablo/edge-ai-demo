import torch
print("PyTorch 版本:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU device count:", torch.cuda.device_count())
    print("Current device:", torch.cuda.current_device())
    
x = torch.rand(3, 3)
print(x)
print("FFT:", torch.fft.fft(x))

