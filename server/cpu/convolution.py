def cpu_convolution(input_tensor, kernel, bias):
    H, W, C_in = input_tensor.shape
    kH, kW, C_in_k, C_out = kernel.shape

    # Validate input dimensions
    assert C_in == C_in_k, "Kernel depth does not match input channels!"

    # Padding
    pad_h, pad_w = kH // 2, kW // 2
    padded_input = np.pad(input_tensor, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant')

    # Output tensor
    output_tensor = np.zeros((H, W, C_out), dtype=np.float32)

    # Convolution
    for h in range(H):
        for w in range(W):
            for oc in range(C_out):
                val = 0.0
                for kh in range(kH):
                    for kw in range(kW):
                        for ic in range(C_in):
                            val += (
                                padded_input[h + kh, w + kw, ic]
                                * kernel[kh, kw, ic, oc]
                            )
                output_tensor[h, w, oc] = val + bias[oc]
    return output_tensor