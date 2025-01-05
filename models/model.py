import typing as tp
import torch
from models import modules as m
from vector_quantize_pytorch import VectorQuantize, ResidualVQ
from torch import nn


def Scale_Normalize(sig):
    scale = torch.max(sig)
    sig_norm = sig / scale
    return scale, sig_norm


class ECGCodec(nn.Module):
    def __init__(self,
                 encoder: m.SEANetEncoder,
                 decoder: m.SEANetDecoder,
                 vector_quantizer,
                 frame_length,
                 channels: int = 1,
                 normalize: bool = False,
                 overlap: float = 0.01,
                 bins: int = 1024,
                 dims: int = 128
                 ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vq = vector_quantizer
        self.bins = bins
        # self.sample_rate = sample_rate
        # self.segment = segment
        self.channels = channels
        self.overlap = overlap
        self.normalize = normalize
        self.training = True
        # self.frame_length = sample_rate * segment
        self.frame_length = frame_length
    @property
    def segment_length(self) -> tp.Optional[int]:
        if self.segment is None:
            return None
        return int(self.segment * self.sample_rate)

    @property
    def segment_stride(self) -> tp.Optional[int]:
        segment_length = self.segment_length
        if segment_length is None:
            return None
        return max(1, int((1 - self.overlap) * segment_length))

    def encode(self, x: torch.Tensor):
        emb = self.encoder(x)

        indices = 0

        if self.vq is not None:
            emb = emb.transpose(1, 2)
            codes, indices, embedding_loss = self.vq(emb)
            embedding_loss = torch.sum(embedding_loss)
        else:
            codes = emb.transpose(1, 2)
            embedding_loss = 0

        codes = codes.transpose(1, 2)

        return codes, indices, embedding_loss

    def decode(self, codes, use_indices=False):
        if use_indices:
            encoded_frame = codes.transpose(0, 1)
            emb = self.vq.get_output_from_indices(encoded_frame)
        else:
            emb = codes
        out = self.decoder(emb)
        return out

    def compress(self, x, use_indices=False):
        assert x.dim() == 3
        _, channels, length = x.shape
        segment_length = self.frame_length
        stride = self.frame_length

        encoded_frames = []

        for offset in range(0, length, stride):
            frame = x[:, :, offset: offset + segment_length]
            encoded_frame, indices, _ = self.encode(frame)
            if use_indices:
                encoded_frames.append(indices)
            else:
                encoded_frames.append(encoded_frame)
        return encoded_frames

    def decompress(self, encoded_frames, use_indices=False):
        stride = self.frame_length
        frames = [self.decode(frame, use_indices) for frame in encoded_frames]

        device = frames[0].device
        dtype = frames[0].dtype
        shape = frames[0].shape[:-1]

        last_frame_length = frames[-1].shape[-1]
        total_size = stride * (len(frames) - 1) + last_frame_length

        out = torch.zeros(*shape, total_size, device=device, dtype=dtype)
        offset: int = 0
        count = 0
        for frame in frames:
            if count == len(frames) - 1:
                out[..., offset:offset + last_frame_length] += frame[:, :, :last_frame_length]
            else:
                out[..., offset:offset + self.frame_length] += frame[:, :, :self.frame_length]
                offset += stride
                count += 1
        return out

    def forward(self, x: torch.Tensor):
        frames, indices, embedding_loss = self.encode(x)
        x_hat = self.decode(frames)[:, :, :x.shape[-1]]
        return embedding_loss, x_hat, indices

    @staticmethod
    def get_exp_model(ratios=[8,5,4,2], e_dims=1024, codebook_dims=16, codebook_size=1024, n_q=4, use_lookup=True):
        channels = 1
        frame_length = 10800

        norm = 'weight_norm'
        encoder = m.SEANetEncoder(ratios=ratios, channels=channels, norm=norm, causal=True, dimension=e_dims)
        decoder = m.SEANetDecoder(ratios=ratios, channels=channels, norm=norm, causal=True, dimension=e_dims)

        if use_lookup:
            # Use STE to back propagate gradients
            rvq = ResidualVQ(
                dim=e_dims,
                codebook_size=codebook_size,
                codebook_dim=codebook_dims,
                num_quantizers=n_q,
                commitment_weight=0.25,
                learnable_codebook=True,
                ema_update=False
            )
        else:
            # Use EMA for codebook update
            rvq = ResidualVQ(
                dim=e_dims,
                codebook_size=codebook_size,
                num_quantizers=n_q,
                decay=0.8
            )

        model = ECGCodec(
            encoder=encoder,
            decoder=decoder,
            vector_quantizer=rvq,
            frame_length=frame_length,
            bins=codebook_size
        )

        return model
