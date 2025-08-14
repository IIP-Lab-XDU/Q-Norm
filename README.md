# Q-Norm
[ICCV'25] Q-Norm
## Getting Started

### (1) Get image quality feature

Refer to the official documentation of CONTRIQUE [here](https://github.com/pavancm/CONTRIQUE) and download the weights

You can also use other quality representations by other advanced BIQA models, but you need to modify the feature dimensions to make them match.

```
from CONTRIQUE_model import CONTRIQUE_model
from torchvision import transforms
import torch

b, c, h, w = x.shape
image_2 = transforms.Resize([h // 2, w // 2])(x)
quality_model = CONTRIQUE_model(models.resnet50(pretrained=False), 2048)
quality_model.load_state_dict(torch.load('/data2/zln/ultralytics_qnorm/qNorm/CONTRIQUE_checkpoint25.tar'))
device=x.device
quality_model = quality_model.to(device)
quality_model.eval()
_, _, _, _, model_feat, model_feat_2, _, _ = quality_model(x, image_2)  # model_feat:2,2048
quality = torch.hstack((model_feat, model_feat_2))  # quality:2,4096
```

### (2) Integrate Q-Norm into the model

```
from qnorm import QualityNorm
qn=QualityNorm(num_features=in_channel)
```

