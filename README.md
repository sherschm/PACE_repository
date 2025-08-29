# PACE Modelling & control code

## Requirements

- Python 3.8+ (recommended)
- `venv` (comes with Python)

## Setup Instructions

1. **OPen a command prompt and clone the repository. Then, make a Python virtual environment**
    ```bash
    git clone https://git.ccfe.ac.uk/pace-training-material/modeling-and-control-notebooks.git
    cd modeling-and-control-notebooks
    python -m venv venv 
    ```

2. **Activate virtual environment and import the required python packages**
- **Windows**
In a command prompt run:

```shell
venv\Scripts\activate.bat
```

- **Linux**

```shell
source venv/bin/activate
```

3. **Install dependencies**
```shell
pip install --upgrade pip
pip install -r requirements.txt
```