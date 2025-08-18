If wanting to add dependancies for Fixed:

from google.colab import drive
drive.mount('/content/drive')
!pip install -U datasets huggingface_hub git-lfs
!git lfs install
%cd /content/drive/MyDrive
!git clone https://huggingface.co/datasets/VQA-Illusion/IllusionAnimals_train


For precomputed, use:

from google.colab import drive
drive.mount('/content/drive')
!pip install -U datasets huggingface_hub git-lfs
!git lfs install
%cd /content/drive/MyDrive
!git clone https://huggingface.co/datasets/VQA-Illusion/IllusionAnimals_train
from datasets import load_dataset

dataset = load_dataset(
    "imagefolder",
    data_dir="/content/drive/MyDrive/IllusionAnimals_train",
    split="train"
)
print(dataset)
