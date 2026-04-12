def conv2d(image, kernel, stride=1, padding=0):
    """
    Apply 2D convolution to a single-channel image.
    """
    # Write code here
    H = len(image)
    W = len(image[0])

    k_h = len(kernel)
    k_w = len(kernel[0])

    padded_H = H + 2 * padding
    padded_W = W + 2 * padding
    padded_image = [[0.0 for _ in range(padded_W)] for _ in range(padded_H)]
    for i in range(H):
        for j in range(W):
            padded_image[i + padding][j + padding] = float(image[i][j])

    H_out = (padded_H - k_h) // stride + 1
    W_out = (padded_W - k_w) // stride + 1
    output_image = [[0.0 for _ in range(W_out)] for _ in range(H_out)]
    for i in range(H_out):
        for j in range(W_out):
            r = i * stride
            c = j * stride
            acc = 0.0
            for m in range(k_h):
                for n in range(k_w):
                    acc += padded_image[r + m][c + n] * kernel[m][n]
            output_image[i][j] = acc

    return output_image