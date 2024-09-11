## Get Started

```bash
# Clone the Repository
git clone https://github.com/DorinDaniil/Garage.git

# Create Virtual Environment with Conda
conda create --name garage python=3.10
conda activate garage

# Install Dependencies
pip install -r requirements.txt
pip install controlnet_aux==0.0.5

# Download the Required Weights
git lfs clone https://huggingface.co/JunhaoZhuang/PowerPaint-v1/ Padd/models/checkpoints/ppt-v1 
```
