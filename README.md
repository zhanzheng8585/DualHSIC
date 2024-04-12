# This codebase contains the implementation of **[DualHSIC: HSIC-Bottleneck and Alignment for Continual Learning](https://proceedings.mlr.press/v202/wang23ar.html) (ICML 2023)**.

Based on [https://github.com/aimagelab/mammoth](Mammoth)

The command for reproducing the results on Split CIFAR-100 with DualHSIC + DER++:
```
bash scripts/DualHSIC_derpp.sh
```
The arguments are self-explanatory and correspond to hyperparameter settings presented in our paper.

## Cite
```
@inproceedings{wang2023dualhsic,
  title={DualHSIC: HSIC-bottleneck and alignment for continual learning},
  author={Wang, Zifeng and Zhan, Zheng and Gong, Yifan and Shao, Yucai and Ioannidis, Stratis and Wang, Yanzhi and Dy, Jennifer},
  booktitle={International Conference on Machine Learning},
  pages={36578--36592},
  year={2023},
  organization={PMLR}
}
