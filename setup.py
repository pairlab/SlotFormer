from setuptools import setup, find_packages

requirements = [
    'numpy', 'pyyaml', 'six', 'tqdm', 'opencv-python', 'matplotlib', 'open3d',
    'wandb', 'moviepy', 'imageio', 'torch', 'torchvision', 'torchmetrics',
    'pytorch-lightning', 'pycocotools', 'pandas', 'einops', 'phyre', 'nerv'
]


setup(
    name="slotformer",
    version='0.1.0',
    description="Learning Object-Centric Dynamics Model with SlotFormer",
    long_description="Learning Object-Centric Dynamics Model with SlotFormer",
    author="Ziyi Wu",
    author_email="ziyiwu@cs.toronto.edu",
    license="MIT",
    url="https://arxiv.org/abs/2210.05861",
    keywords="object-centric dynamics model",
    packages=find_packages(),
    install_requires=requirements,
)
