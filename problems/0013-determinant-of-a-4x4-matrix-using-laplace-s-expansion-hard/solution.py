import numpy as np

def determinant_4x4(matrix: list[list[int|float]]) -> float:
    
    matrix = np.asarray(matrix, dtype = float)
    n = matrix.shape[0]

    # base case 1 x 1
    if n == 1:
        return float(matrix[0,0])

    #base  case 2 x 2

    if n == 2:
        return float(
            matrix[0,0] * matrix[1,1]
            - matrix[0,1] * matrix[1,0]
        )

    det = 0.0

    # exopantion along the first row 
    for j in range(n):

        # Remove first row and selected column
        minor = np.delete(
            np.delete(matrix, 0, axis = 0),
            j , axis = 1
        )


        # recursive Lapllace expansion 
        det += (
            (-1) ** j * matrix[0, j] * determinant_4x4(minor)
        )
    
    return float(det)



