import numpy as np
import control as ct

def clean_transfer_function(G, tol_factor=1e-10):
    """
    Clean up a transfer function by normalizing and truncating
    small coefficients below a tolerance.
    
    Parameters
    ----------
    G : ct.TransferFunction
        Transfer function to clean.
    tol_factor : float, optional
        Relative tolerance factor (default = 1e-12).
    
    Returns
    -------
    G_clean : ct.TransferFunction
        Transfer function with tiny coefficients set to zero.
    """
    # extract numerator and denominator arrays
    num, den = G.num[0][0].copy(), G.den[0][0].copy()
    
    # compute tolerance based on largest coefficient
    maxcoeff = max(np.max(np.abs(num)), np.max(np.abs(den)))
    tol = tol_factor * maxcoeff
    
    # zero out small coefficients
    num[np.abs(num) < tol] = 0
    den[np.abs(den) < tol] = 0
    
    # return cleaned transfer function
    return ct.TransferFunction(num, den)