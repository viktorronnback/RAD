# RAD - Realistic Anonymization using Diffusion
RAD is a full-body realistic anonymization pipeline based on Stable Diffusion.

![ezgif com-resize (12)](https://github.com/viktorronnback/realistic-anonymization-test/assets/134917172/2f7a31fb-5058-4c4b-9a8d-43c984b48ea6)
![ezgif com-resize (8)](https://github.com/viktorronnback/realistic-anonymization-test/assets/134917172/db521588-acdf-45d6-9755-363f8ae6d119)
![ezgif com-crop (1)](https://github.com/viktorronnback/realistic-anonymization-test/assets/134917172/f689d8d9-a08a-439f-ac77-7415aa32029a)
![ezgif com-resize (10)](https://github.com/viktorronnback/realistic-anonymization-test/assets/134917172/169665ca-e4d3-4d06-bed8-ac6295c95160)
![ezgif com-resize (11)](https://github.com/viktorronnback/realistic-anonymization-test/assets/134917172/623b4df9-cda1-43cc-b2c5-72e4a4077ca5)

![text-pipe](https://github.com/viktorronnback/realistic-anonymization-test/assets/134917172/8c721ac9-68c8-45dd-aafc-c0921eb1676f)

## Installation

1. Install PyTorch, [instructions](https://pytorch.org/get-started/locally/)

2. Clone repository:
```bash
git clone git@github.com:viktorronnback/realistic-anonymization-test.git realistic-anonymization
```

3. Go to the root directory of the repository:
```bash
cd realistic-anonymization
```

4. Install pip dependencies:
```bash
pip install -r requirements.txt
```

## Run anonymization

Demo anonymization:
```bash
python main.py config.yaml
```

Will automatically download models on the initial run (these can be large >10 GB)

Demo images in anonymizer/input/demo/ are stock-photos from pexels.com.  

## Publication

The tool was created as part of a master thesis in Computer Science at Linköping University.

[Link to thesis](#) - RAD: Realistic Anonymization of Images using Stable Diffusion
