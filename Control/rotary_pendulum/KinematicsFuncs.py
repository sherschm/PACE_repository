import sympy as sp
from sympy import Matrix
# Rotation matrix function: assuming yaw-pitch rotation (Z-Y axes)

def Rz(θ):
        return Matrix([
        [sp.cos(θ), -sp.sin(θ), 0],
        [sp.sin(θ),  sp.cos(θ), 0],
        [0, 0, 1]
    ])

def Ry(θ):
        return Matrix([
        [sp.cos(θ), 0, sp.sin(θ)],
        [0, 1, 0],
        [-sp.sin(θ), 0, sp.cos(θ)]
    ])

def calc_R20(θ1, θ2):
    return Ry(θ2) *  Rz(θ1)