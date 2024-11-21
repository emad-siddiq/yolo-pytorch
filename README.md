# YOLO Implementation in PyTorch

### You Only Look Once: Unified, Real-Time Object Detection
https://arxiv.org/abs/1506.02640

## Setup Instructions

### Local Development

1. Clone the repository:
```bash
git clone https://github.com/yourusername/yolo-pytorch.git
cd yolo-pytorch
```

2. Create Anaconda environment:
```bash
conda env create -f environment.yml
conda activate yolo-pytorch
```

3. Install additional requirements:
```bash
pip install -r requirements.txt
```

### Google Colab Development

1. Clone your repository in Colab:
```python
!git clone https://github.com/yourusername/yolo-pytorch.git
%cd yolo-pytorch
```

2. Install requirements:
```python
!pip install -r requirements.txt
```

3. Add repository to Python path:
```python
import sys
sys.path.append('/content/yolo-pytorch')
```

### Git Integration with Google Colab

1. Set up Git credentials in Colab:
```python
!git config --global user.email "your.email@example.com"
!git config --global user.name "Your Name"
```

2. To save changes back to GitHub:
```python
!git add .
!git commit -m "Update from Colab"
!git push origin main
```