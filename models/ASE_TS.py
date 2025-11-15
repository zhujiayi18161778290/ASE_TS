import torch
import torch.nn as nn
from layers.RevIN import RevIN



class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class ComplexLinear(nn.Module):
    def __init__(self, seq_len_in, pred_len_out):
        super().__init__()
        self.real = nn.Linear(seq_len_in, pred_len_out)
        self.imag = nn.Linear(seq_len_in, pred_len_out)

    def forward(self, x):  # x: [B, C, T]
        # x.real / x.imag: [B, C, T]
        real_out = self.real(x.real) - self.imag(x.imag)  # [B, C, T_out]
        imag_out = self.real(x.imag) + self.imag(x.real)  # [B, C, T_out]
        return torch.complex(real_out, imag_out)  # [B, C, T_out]

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.hidden_size = configs.hidden_size
        self.use_decomposition = configs.use_decomposition
        self.revin_layer = RevIN(configs.enc_in, affine=True, subtract_last=False)

        kernel_size = 25
        self.decomposition = series_decomp(kernel_size)
     
        self.linear = nn.Linear(int(self.seq_len / 2) + 1, int(self.seq_len / 2) + 1).to(torch.cfloat)
        self.complex_linear = ComplexLinear(int(self.seq_len / 2) + 1, int(self.pred_len / 2) + 1)
        
        # Replace GELU with LeakyReLU
        self.linear_residual = nn.Sequential(
            nn.Linear(self.seq_len, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.pred_len)
        )

        self.linear_trend = nn.Sequential(
            nn.Linear(self.seq_len, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.pred_len)
        )

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec, mask=None):
        B, L, C = x.size()

        # RevIN
        z = x
        z = self.revin_layer(z, 'norm')
        x = z
        
        # FFT
        x_fft = torch.fft.rfft(x, dim=1)   # domain conversion

        # Adaptive Spectral Enhancement
        x_fft_adjusted = self.linear(x_fft.permute(0, 2, 1)).permute(0, 2, 1)# flip the spectrum
        x_fft_enhanced = x_fft + x_fft_adjusted
        
        # IFFT
        x_enhanced = torch.fft.irfft(x_fft_enhanced, dim=1)

        # Forecaster
        if self.use_decomposition:
            residual, trend = self.decomposition(x_enhanced)
            residual = self.linear_residual(residual.permute(0, 2, 1)).permute(0, 2, 1)
            trend = self.linear_trend(trend.permute(0, 2, 1)).permute(0, 2, 1)
            forecasted = residual + trend
        else:
            forecasted = self.linear_residual(x_enhanced.permute(0, 2, 1)).permute(0, 2, 1)

        # FFT
        forecasted_fft = torch.fft.rfft(forecasted, dim=1)
        
        # Spectral Restoration
        x_fft_inverse = self.complex_linear(x_fft_adjusted.permute(0, 2, 1)).permute(0, 2, 1)
        out_fft = forecasted_fft - x_fft_inverse
        
        # IFFT
        out = torch.fft.irfft(out_fft, dim=1)

        # inverse RevIN
        z = out
        z = self.revin_layer(z, 'denorm')
        out = z

        return out
