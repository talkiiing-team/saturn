import numpy as np
from numpy.typing import ArrayLike

from saturn.libs.broker import broker

model = None
if broker.is_worker_process:
    import torch
    import torch.nn as nn
    import torchvision.models.video as video

    class I3DModel(nn.Module):
        def __init__(self, num_classes=400, pretrained=True):
            super(I3DModel, self).__init__()
            self.i3d = video.swin3d_t(video.swin_transformer.Swin3D_T_Weights)
            self.i3d.head = nn.Flatten()

        def forward(self, x):
            return self.i3d(x)

    model = I3DModel(pretrained=True).to("cuda")
    model.eval()


def extract_video_embedding(imgs_list: list[ArrayLike]):
    """
    Функция для извлечения эмбеддингов из видео.
    """

    video_tensor = (
        torch.FloatTensor(np.expand_dims(np.stack(imgs_list), 0))
        .permute(0, 4, 1, 2, 3)
        .contiguous()
        .to("cuda")
    )

    with torch.no_grad():
        embeddings = model(video_tensor)

    return embeddings.cpu().detach().numpy()[0].tolist()
