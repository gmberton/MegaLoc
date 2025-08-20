# MegaLoc
An image retrieval model for any localization task, which achieves SOTA on most VPR datasets, including indoor and outdoor ones.

[Gradio Demo](https://ef3588c578029c6fdd.gradio.live/) - [ArXiv](https://arxiv.org/abs/2502.17237) - [Paper on ArXiv](https://arxiv.org/pdf/2502.17237) - [Paper on HF](https://huggingface.co/papers/2502.17237) - [Model on HF](https://huggingface.co/gberton/MegaLoc).

### Demo
Try the demo on your own images to see how good MegaLoc is! The demo uses a database of ~5M street-view images from San Francisco, and when you upload one it will find the most similar one from the same place.

<img width="746" height="576" alt="image" src="https://github.com/user-attachments/assets/4e7a3eec-dfee-4aae-83cc-f5146a1b421d" />


### Using the model
You can use the model with torch.hub, as simple as this
```
import torch
model = torch.hub.load("gmberton/MegaLoc", "get_trained_model")
```

For more complex uses, like computing results on VPR datasets, visualizing predictions and so on, you can use our [VPR-methods-evaluation](https://github.com/gmberton/VPR-methods-evaluation), which lets you do all this for MegaLoc and multiple other VPR methods on labelled or unlabelled datasets.

### Qualitative examples
Here are some examples of top-1 retrieved images from the SF-XL test set, which has 2.8M images as database.

![teaser](https://github.com/user-attachments/assets/a90b8d4c-ab53-4151-aacc-93493d583713)



## Acknowledgements / Cite / BibTex

If you use this repository please cite the following
```
@misc{berton_2025_megaloc,
      title={MegaLoc: One Retrieval to Place Them All}, 
      author={Gabriele Berton and Carlo Masone},
      year={2025},
      eprint={2502.17237},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2502.17237}, 
}
```

