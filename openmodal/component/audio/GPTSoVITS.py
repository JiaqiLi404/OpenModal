import contextlib
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.nn import functional as F

from openmodal.block import attentions, MultiHeadAttention
from openmodal.block.audio.audio_generator import AudioGenerator
from openmodal.block.audio.multi_reference_timber_encoder import MRTE
from openmodal.block.audio.posterior_encoder import PosteriorEncoder
from openmodal.block.audio.residual_vector_quantizer import ResidualVectorQuantizer
from openmodal.block.conv import Conv1dGLU
from openmodal.block.coupling_block import ResidualCouplingBlock
from openmodal.block.functions import Mish
from openmodal.block.normalizations import LinearNorm
from openmodal.util.torch import sequence_mask, rand_slice_segments


class TextEncoder(nn.Module):
    def __init__(
            self,
            n_vocab,
            out_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            latent_channels=192,
    ):
        super().__init__()
        self.n_vocab=n_vocab
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.latent_channels = latent_channels

        self.text_embedding = nn.Embedding(n_vocab, hidden_channels)
        nn.init.normal_(self.text_embedding.weight, 0.0, hidden_channels ** -0.5)

        self.ssl_proj = nn.Conv1d(768, hidden_channels, 1)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

        self.encoder_ssl = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers // 2,
            kernel_size,
            p_dropout,
            flow_version="layer"
        )

        self.encoder_text = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            flow_version="layer"
        )


        self.mrte = MRTE()

        self.encoder2 = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers // 2,
            kernel_size,
            p_dropout,
            flow_version="layer"
        )



    def forward(self, y, y_lengths, text, text_lengths, ge, speed=1, test=None):
        y_mask = torch.unsqueeze(sequence_mask(y_lengths, y.size(2)), 1).to(
            y.dtype
        )

        y = self.ssl_proj(y * y_mask) * y_mask

        y = self.encoder_ssl(y * y_mask, y_mask)

        text_mask = torch.unsqueeze(
            sequence_mask(text_lengths, text.size(1)), 1
        ).to(y.dtype)
        if test == 1:
            text[:, :] = 0
        text = self.text_embedding(text).transpose(1, 2)
        text = self.encoder_text(text * text_mask, text_mask)
        y = self.mrte(y, y_mask, text, text_mask, ge)
        y = self.encoder2(y * y_mask, y_mask)
        if (speed != 1):
            y = F.interpolate(y, size=int(y.shape[-1] / speed) + 1, mode="linear")
            y_mask = F.interpolate(y_mask, size=y.shape[-1], mode="nearest")
        stats = self.proj(y) * y_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        return y, m, logs, y_mask

    def extract_latent(self, x):
        x = self.ssl_proj(x)
        quantized, codes, commit_loss, quantized_list = self.quantizer(x)
        return codes.transpose(0, 1)

    def decode_latent(self, codes, y_mask, refer, refer_mask, ge):
        quantized = self.quantizer.decode(codes)

        y = self.vq_proj(quantized) * y_mask
        y = self.encoder_ssl(y * y_mask, y_mask)

        y = self.mrte(y, y_mask, refer, refer_mask, ge)

        y = self.encoder2(y * y_mask, y_mask)

        stats = self.proj(y) * y_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        return y, m, logs, y_mask, quantized

class MelStyleEncoder(nn.Module):
    """MelStyleEncoder"""

    def __init__(
        self,
        n_mel_channels=80,
        style_hidden=128,
        style_vector_dim=256,
        style_kernel_size=5,
        style_head=2,
        dropout=0.1,
    ):
        super(MelStyleEncoder, self).__init__()
        self.in_dim = n_mel_channels
        self.hidden_dim = style_hidden
        self.out_dim = style_vector_dim
        self.kernel_size = style_kernel_size
        self.n_head = style_head
        self.dropout = dropout

        self.spectral = nn.Sequential(
            LinearNorm(self.in_dim, self.hidden_dim),
            Mish(),
            nn.Dropout(self.dropout),
            LinearNorm(self.hidden_dim, self.hidden_dim),
            Mish(),
            nn.Dropout(self.dropout),
        )

        self.temporal = nn.Sequential(
            Conv1dGLU(self.hidden_dim, self.hidden_dim, self.kernel_size, self.dropout),
            Conv1dGLU(self.hidden_dim, self.hidden_dim, self.kernel_size, self.dropout),
        )

        self.slf_attn = MultiHeadAttention(
            self.n_head,
            self.hidden_dim,
            self.hidden_dim // self.n_head,
            self.hidden_dim // self.n_head,
            self.dropout,
        )

        self.fc = LinearNorm(self.hidden_dim, self.out_dim)

    def temporal_avg_pool(self, x, mask=None):
        if mask is None:
            out = torch.mean(x, dim=1)
        else:
            len_ = (~mask).sum(dim=1).unsqueeze(1)
            x = x.masked_fill(mask.unsqueeze(-1), 0)
            x = x.sum(dim=1)
            out = torch.div(x, len_)
        return out

    def forward(self, x, mask=None):
        x = x.transpose(1, 2)
        if mask is not None:
            mask = (mask.int() == 0).squeeze(1)
        max_len = x.shape[1]
        slf_attn_mask = (
            mask.unsqueeze(1).expand(-1, max_len, -1) if mask is not None else None
        )

        # spectral
        x = self.spectral(x)
        # temporal
        x = x.transpose(1, 2)
        x = self.temporal(x)
        x = x.transpose(1, 2)
        # self-attention
        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1), 0)
        x, _ = self.slf_attn(x, mask=slf_attn_mask)
        # fc
        x = self.fc(x)
        # temoral average pooling
        w = self.temporal_avg_pool(x, mask=mask)

        return w.unsqueeze(-1)

class GPTSoVITS(nn.Module):
    """
    Synthesizer for Training
    """

    def __init__(
        self,
        n_vocab,
        spec_channels,
        segment_size,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        n_speakers=0,
        gin_channels=0,
        use_sdp=True,
        semantic_frame_rate=None,
        freeze_quantizer=None,
        **kwargs
    ):
        super().__init__()
        self.n_vocab = n_vocab
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels

        self.use_sdp = use_sdp
        self.enc_p = TextEncoder(
            n_vocab,
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
        )
        self.dec = AudioGenerator(
            inter_channels,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=gin_channels,
        )
        self.enc_q = PosteriorEncoder(
            spec_channels,
            inter_channels,
            hidden_channels,
            5,
            1,
            16,
            gin_channels=gin_channels,
        )
        self.flow = ResidualCouplingBlock(
            inter_channels, hidden_channels, 5, 1, 4, gin_channels=gin_channels
        )

        self.ref_enc =MelStyleEncoder(704, style_vector_dim=gin_channels)

        ssl_dim = 768
        assert semantic_frame_rate in ["25hz", "50hz"]
        self.semantic_frame_rate = semantic_frame_rate
        if semantic_frame_rate == "25hz":
            self.ssl_proj = nn.Conv1d(ssl_dim, ssl_dim, 2, stride=2)
        else:
            self.ssl_proj = nn.Conv1d(ssl_dim, ssl_dim, 1, stride=1)

        self.quantizer = ResidualVectorQuantizer(dimension=ssl_dim, n_q=1, bins=1024)
        self.freeze_quantizer = freeze_quantizer

    def forward(self, ssl, y, y_lengths, text, text_lengths):
        y_mask = torch.unsqueeze(sequence_mask(y_lengths, y.size(2)), 1).to(
            y.dtype
        )
        if(self.version=="v1"):
            ge = self.ref_enc(y * y_mask, y_mask)
        else:
            ge = self.ref_enc(y[:,:704] * y_mask, y_mask)
        with autocast(enabled=False):
            maybe_no_grad = torch.no_grad() if self.freeze_quantizer else contextlib.nullcontext()
            with maybe_no_grad:
                if self.freeze_quantizer:
                    self.ssl_proj.eval()
                    self.quantizer.eval()
            ssl = self.ssl_proj(ssl)
            quantized, codes, commit_loss, quantized_list = self.quantizer(
                ssl, layers=[0]
            )

        if self.semantic_frame_rate == "25hz":
            quantized = F.interpolate(
                quantized, size=int(quantized.shape[-1] * 2), mode="nearest"
            )

        x, m_p, logs_p, y_mask = self.enc_p(
            quantized, y_lengths, text, text_lengths, ge
        )
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=ge)
        z_p = self.flow(z, y_mask, g=ge)

        z_slice, ids_slice = rand_slice_segments(
            z, y_lengths, self.segment_size
        )
        o = self.dec(z_slice, g=ge)
        return (
            o,
            commit_loss,
            ids_slice,
            y_mask,
            y_mask,
            (z, z_p, m_p, logs_p, m_q, logs_q),
            quantized,
        )

    def infer(self, ssl, y, y_lengths, text, text_lengths, test=None, noise_scale=0.5):
        y_mask = torch.unsqueeze(sequence_mask(y_lengths, y.size(2)), 1).to(
            y.dtype
        )
        if(self.version=="v1"):
            ge = self.ref_enc(y * y_mask, y_mask)
        else:
            ge = self.ref_enc(y[:,:704] * y_mask, y_mask)

        ssl = self.ssl_proj(ssl)
        quantized, codes, commit_loss, _ = self.quantizer(ssl, layers=[0])
        if self.semantic_frame_rate == "25hz":
            quantized = F.interpolate(
                quantized, size=int(quantized.shape[-1] * 2), mode="nearest"
            )

        x, m_p, logs_p, y_mask = self.enc_p(
            quantized, y_lengths, text, text_lengths, ge, test=test
        )
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale

        z = self.flow(z_p, y_mask, g=ge, reverse=True)

        o = self.dec((z * y_mask)[:, :, :], g=ge)
        return o, y_mask, (z, z_p, m_p, logs_p)

    @torch.no_grad()
    def decode(self, codes, text, refer, noise_scale=0.5,speed=1):
        def get_ge(refer):
            ge = None
            if refer is not None:
                refer_lengths = torch.LongTensor([refer.size(2)]).to(refer.device)
                refer_mask = torch.unsqueeze(
                    sequence_mask(refer_lengths, refer.size(2)), 1
                ).to(refer.dtype)
                if (self.version == "v1"):
                    ge = self.ref_enc(refer * refer_mask, refer_mask)
                else:
                    ge = self.ref_enc(refer[:, :704] * refer_mask, refer_mask)
            return ge
        if(type(refer)==list):
            ges=[]
            for _refer in refer:
                ge=get_ge(_refer)
                ges.append(ge)
            ge=torch.stack(ges,0).mean(0)
        else:
            ge=get_ge(refer)

        y_lengths = torch.LongTensor([codes.size(2) * 2]).to(codes.device)
        text_lengths = torch.LongTensor([text.size(-1)]).to(text.device)

        quantized = self.quantizer.decode(codes)
        if self.semantic_frame_rate == "25hz":
            quantized = F.interpolate(
                quantized, size=int(quantized.shape[-1] * 2), mode="nearest"
            )
        x, m_p, logs_p, y_mask = self.enc_p(
            quantized, y_lengths, text, text_lengths, ge,speed
        )
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale

        z = self.flow(z_p, y_mask, g=ge, reverse=True)

        o = self.dec((z * y_mask)[:, :, :], g=ge)
        return o

    def extract_latent(self, x):
        ssl = self.ssl_proj(x)
        quantized, codes, commit_loss, quantized_list = self.quantizer(ssl)
        return codes.transpose(0, 1)
