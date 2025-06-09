#!/usr/bin/env python3
import sys
import gguf
import numpy as np

def save_conv2d_layer(f, gguf_writer, prefix, inp_c, filters, size, batch_normalize=True):
    biases = np.fromfile(f, dtype=np.float32, count=filters)
    gguf_writer.add_tensor(prefix + "_biases", biases, raw_shape=(1, filters, 1, 1))

    if batch_normalize:
        scales = np.fromfile(f, dtype=np.float32, count=filters)
        gguf_writer.add_tensor(prefix + "_scales", scales, raw_shape=(1, filters, 1, 1))
        rolling_mean = np.fromfile(f, dtype=np.float32, count=filters)
        gguf_writer.add_tensor(prefix + "_rolling_mean", rolling_mean, raw_shape=(1, filters, 1, 1))
        rolling_variance = np.fromfile(f, dtype=np.float32, count=filters)
        gguf_writer.add_tensor(prefix + "_rolling_variance", rolling_variance, raw_shape=(1, filters, 1, 1))

    weights_count = filters * inp_c * size * size
    weights = np.fromfile(f, dtype=np.float32, count=weights_count)
    ## ggml doesn't support f32 convolution yet, use f16 instead
    weights = weights.astype(np.float16)
    gguf_writer.add_tensor(prefix + "_weights", weights, raw_shape=(filters, inp_c, size, size))


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: %s <yolov4-tiny.weights>" % sys.argv[0])
        sys.exit(1)
    outfile = 'yolov4-tiny.gguf'
    gguf_writer = gguf.GGUFWriter(outfile, 'yolov4-tiny')

    f = open(sys.argv[1], 'rb')
    f.read(20) # skip header
    save_conv2d_layer(f, gguf_writer, "l0", 3, 32, 3)
    save_conv2d_layer(f, gguf_writer, "l1", 32, 64, 3)
    save_conv2d_layer(f, gguf_writer, "l2", 64, 64, 3)
    save_conv2d_layer(f, gguf_writer, "l3", 32, 32, 3)
    save_conv2d_layer(f, gguf_writer, "l4", 32, 32, 3)
    save_conv2d_layer(f, gguf_writer, "l5", 64, 64, 1)
    save_conv2d_layer(f, gguf_writer, "l6", 128, 128, 3)
    save_conv2d_layer(f, gguf_writer, "l7", 64, 64, 3)
    save_conv2d_layer(f, gguf_writer, "l8", 64, 64, 3)
    save_conv2d_layer(f, gguf_writer, "l9", 128, 128, 1)
    save_conv2d_layer(f, gguf_writer, "l10", 256, 256, 3)
    save_conv2d_layer(f, gguf_writer, "l11", 128, 128, 3)
    save_conv2d_layer(f, gguf_writer, "l12", 128, 128, 3)
    save_conv2d_layer(f, gguf_writer, "l13", 256, 256, 1)
    save_conv2d_layer(f, gguf_writer, "l14", 512, 512, 3) #26
    save_conv2d_layer(f, gguf_writer, "l15", 512, 256, 1) #27
    save_conv2d_layer(f, gguf_writer, "l16", 256, 512, 3) #28
    save_conv2d_layer(f, gguf_writer, "l17", 512, 255, 1, batch_normalize=False) #29
    save_conv2d_layer(f, gguf_writer, "l18", 256, 128, 1)
    save_conv2d_layer(f, gguf_writer, "l19", 384, 256, 3)
    save_conv2d_layer(f, gguf_writer, "l20", 256, 255, 1, batch_normalize=False)
    f.close()

    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()
    gguf_writer.close()
    print("{} converted to {}".format(sys.argv[1], outfile))
