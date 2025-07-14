import numpy as np


def RBF(S, Y, flag):
    """
    Computes the parameters of the radial basis function interpolant.

    Parameters:
    S : numpy.ndarray
        Sample site matrix (m x d), where m is the number of sample sites and d is the dimension.
    Y : numpy.ndarray
        Objective function values corresponding to points in S.
    flag : str
        Type of RBF model to use ('cubic', 'TPS', or 'linear').

    Returns:
    lambda : numpy.ndarray
        Vector of RBF model parameters.
    gamma : numpy.ndarray
        Vector of RBF model parameters.
    """
    m, n = S.shape
    P = np.hstack((S, np.ones((m, 1))))  # Add a column of ones for the linear polynomial term

    # Compute the pairwise distance matrix R
    R = np.zeros((m, m))
    for ii in range(m):
        for jj in range(ii, m):
            R[ii, jj] = np.sum((S[ii, :] - S[jj, :]) ** 2)
            R[jj, ii] = R[ii, jj]
    R = np.sqrt(R)

    # Compute the basis function matrix Phi based on the flag
    if flag == 'cubic':  # Cubic RBF
        Phi = R ** 3
    elif flag == 'TPS':  # Thin plate spline RBF
        R[R == 0] = 1  # Avoid log(0)
        Phi = R ** 2 * np.log(R)
    elif flag == 'linear':  # Linear RBF
        Phi = R
    else:
        raise ValueError("Invalid flag. Supported flags are 'cubic', 'TPS', and 'linear'.")

    # Construct the linear system
    A_top = np.hstack((Phi, P))
    A_bottom = np.hstack((P.T, np.zeros((n + 1, n + 1))))
    A = np.vstack((A_top, A_bottom))

    RHS = np.vstack((Y.reshape(-1, 1), np.zeros((n + 1, 1))))

    # Solve the linear system
    params = np.linalg.lstsq(A, RHS, rcond=None)[0]
    lambda_ = params[:m]
    gamma = params[m:]

    return lambda_, gamma


def RBF_eval(X, S, lambda_, gamma, flag):
    """
    Evaluates the RBF model at points X.

    Parameters:
    X : numpy.ndarray
        Points where function values should be calculated (mX x nX).
    S : numpy.ndarray
        Sample sites where function values are known (mS x nS).
    lambda_ : numpy.ndarray
        RBF model parameters.
    gamma : numpy.ndarray
        Parameters of the optional polynomial tail.
    flag : str
        Type of RBF model to use ('cubic', 'TPS', or 'linear').

    Returns:
    Yest : numpy.ndarray
        Estimated function values at the points in X.
    """
    mX, nX = X.shape
    mS, nS = S.shape

    # Check if dimensions match, transpose if necessary
    if nX != nS:
        X = X.T
        mX, nX = X.shape

    # Compute pairwise distances between points in X and S
    R = np.zeros((mX, mS))
    for ii in range(mX):
        for jj in range(mS):
            R[ii, jj] = np.linalg.norm(X[ii, :] - S[jj, :])

    # Compute the basis function matrix Phi based on the flag
    if flag == 'cubic':  # Cubic RBF
        Phi = R ** 3
    elif flag == 'TPS':  # Thin plate spline RBF
        R[R == 0] = 1  # Avoid log(0)
        Phi = R ** 2 * np.log(R)
    elif flag == 'linear':  # Linear RBF
        Phi = R
    else:
        raise ValueError("Invalid flag. Supported flags are 'cubic', 'TPS', and 'linear'.")

    # Compute the predicted function values
    Yest1 = Phi @ lambda_  # First part of the response surface
    Yest2 = np.hstack((X, np.ones((mX, 1)))) @ gamma  # Optional polynomial tail
    Yest = Yest1 + Yest2  # Predicted function value

    return Yest


# Example usage
if __name__ == "__main__":
    S = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])  # Sample sites
    Y = np.array([0, 1, 1, 2])  # Objective function values
    flag = 'cubic'  # Type of RBF

    lambda_, gamma = RBF(S, Y, flag)
    print("Lambda:", lambda_)
    print("Gamma:", gamma)