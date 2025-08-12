# PACE Modelling & control code

## Requirements

- Python 3.8+ (recommended)
- `venv` (comes with Python)

## Setup Instructions

1. **Clone the repository and make a Python virtual environment**
    ```bash
    git clone https://git.ccfe.ac.uk/pace-training-material/modeling-and-control-notebooks.git
    cd modeling-and-control-notebooks
    python3 -m venv venv 
    ```

2. **Activate virtual environment and import the required python packages**
- **Windows**

```shell
\Scripts\Activate.ps1
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

If requirements.txt doesn’t exist yet, you can generate it after installing dependencies:

```shell
pip freeze > requirements.txt
```