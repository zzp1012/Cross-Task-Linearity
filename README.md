# Cross-Task-Linearity
Release codes soon. 
"[On the Emergence of Cross-Task Linearity in Pretraining-Finetuning Paradigm](https://arxiv.org/abs/2402.03660)" (accepted by ICML 2024).

## Abstract
The pretraining-finetuning paradigm has become the prevailing trend in modern deep learning. In this work, we discover an intriguing linear phenomenon in models that are initialized from a common pretrained checkpoint and finetuned on different tasks, termed as *Cross-Task Linearity (CTL)*. Specifically, we show that if we linearly interpolate the weights of two finetuned models, the features in the weight-interpolated model are often approximately equal to the linear interpolation of features in two finetuned models at each layer. We provide comprehensive empirical evidence supporting that CTL consistently occurs for finetuned models that start from the same pretrained checkpoint. We conjecture that in the pretraining-finetuning paradigm, neural networks approximately function as linear maps, mapping from the parameter space to the feature space. Based on this viewpoint, our study unveils novel insights into explaining model merging/editing, particularly by translating operations from the parameter space to the feature space. Furthermore, we delve deeper into the underlying factors for the emergence of CTL, highlighting the role of pretraining.

## Code
`task_arithemetic` contains the code for *section 4.3 Insights into Task Arithmetic*, including cross-task-linearity on addition and negation, model stitching on addition and negation for ViT models. `task_arithemetic_t5` contains the code for T5 model in NLP domain.
