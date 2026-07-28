# MegaLoc
An image retrieval model for any localization task, which achieves SOTA on most VPR datasets, including indoor and outdoor ones.

[Gradio Demo](https://gmberton.github.io/demos-url/megaloc/) - [ArXiv](https://arxiv.org/abs/2502.17237) - [Paper on ArXiv](https://arxiv.org/pdf/2502.17237) - [Paper on HF](https://huggingface.co/papers/2502.17237) - [Model on HF](https://huggingface.co/gberton/MegaLoc).

### Demo
Try the demo on your own images to see how good MegaLoc is! The demo uses a database of ~5M street-view images from San Francisco, and when you upload one it will find the most similar one from the same place.

<img width="746" height="576" alt="image" src="https://github.com/user-attachments/assets/4e7a3eec-dfee-4aae-83cc-f5146a1b421d" />


### Using the model
You can use the model with torch.hub, as simple as this
```
import torch
model = torch.hub.load("gmberton/MegaLoc", "get_trained_model")
```

Here is a complete example with the same preprocessing that we use for evaluation (ImageNet normalization, resize to 322x322)

```python
import torchvision.transforms as tfm
from PIL import Image

transform = tfm.Compose([
    tfm.ToTensor(),
    tfm.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    tfm.Resize(size=[322, 322], antialias=True),
])

images = torch.stack([transform(Image.open(path).convert("RGB")) for path in ["im1.jpg", "im2.jpg"]])
with torch.inference_mode():
    descriptors = model(images)  # shape [2, 8448], L2-normalized

similarities = descriptors @ descriptors.T  # cosine similarities
```

For more complex uses, like computing results on VPR datasets, visualizing predictions and so on, you can use our [VPR-methods-evaluation](https://github.com/gmberton/VPR-methods-evaluation), which lets you do all this for MegaLoc and multiple other VPR methods on labelled or unlabelled datasets.

### FAQ

**How do I reproduce the paper's results?** With [VPR-methods-evaluation](https://github.com/gmberton/VPR-methods-evaluation) and the datasets from [VPR-datasets-downloader](https://github.com/gmberton/VPR-datasets-downloader), which has the exact splits we used ([#13](https://github.com/gmberton/MegaLoc/issues/13)).

**Will you release the training code or datasets?** No: the datasets come from sources that we are not allowed to redistribute, and we are not planning to release the training code ([#5](https://github.com/gmberton/MegaLoc/issues/5), [#9](https://github.com/gmberton/MegaLoc/issues/9)).

**Should I fine-tune MegaLoc on my data?** Usually not, unless your data is very out-of-distribution, like caves or underwater ([#6](https://github.com/gmberton/MegaLoc/issues/6)).

**How should I handle 360° images?** Crop each panorama into perspective crops and treat them as independent images ([#10](https://github.com/gmberton/MegaLoc/issues/10)).

### Qualitative examples
Here are some examples of top-1 retrieved images from the SF-XL test set, which has 2.8M images as database.

![teaser](https://github.com/user-attachments/assets/a90b8d4c-ab53-4151-aacc-93493d583713)



## Acknowledgements / Cite / BibTex

If you use this repository please cite the following
```bibtex
@inproceedings{Berton_2025_MegaLoc,
    author    = {Berton, Gabriele and Masone, Carlo},
    title     = {MegaLoc: One Retrieval to Place Them All},
    booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops},
    month     = {June},
    year      = {2025},
    pages     = {2886--2892}
}
```

