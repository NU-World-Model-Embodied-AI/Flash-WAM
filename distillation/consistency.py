"""Consistency-function scalings (Karras boundary conditions)."""
def scalings_for_boundary_conditions(sigma, sigma_data=0.5):
    """
    Consistency boundary condition: f(x, 0) = x.
    c_skip(0)=1, c_out(0)=0 → identity at sigma=0.
    """
    c_skip = sigma_data ** 2 / (sigma ** 2 + sigma_data ** 2)
    c_out = sigma * sigma_data / (sigma ** 2 + sigma_data ** 2) ** 0.5
    return c_skip, c_out
