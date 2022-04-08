from enviroment import *
import caffe
import sys

import serialize_json_8bit
import serialize_json_fp16
import infer_complexity
import unittest_generate_samples

eps = 1e-5

def openCaffeSpec(proto_file):
    net_spec = caffe.proto.caffe_pb2.NetParameter()
    with open(proto_file) as f:
        s = f.read()
        text_format.Merge(s, net_spec)
    return net_spec

def convert_net_json(minidnn_json, prototxt_path, output_layers, mode='cpu', mean_val=0, scale_val=1/256.0):
    net_spec = openCaffeSpec(prototxt_path)
    total_complexity, total_model_size, complexity_dict, model_size_dict, shape_dict = \
        infer_complexity.infer_complexity(prototxt_path, output_layers)
    serialize_json.record_json(net_spec, shape_dict, mean_val, scale_val, mode, minidnn_json)


def get_layer_params(net_params, layer_name, param_name):
    for idx in range(0, len(net_params.layer)):
        if net_params.layer[idx].name == layer_name:
            if param_name == 'quantization_param':
                return net_params.layer[idx].quantization_param

    return

def convert_weights(weights, gpu_file, layer_name):
    n = weights.shape[0]
    c = weights.shape[1]
    h = weights.shape[2]
    w = weights.shape[3]
    if n % 8 != 0:
        print('warning: for GPU models, channel should be divided by 8, check manually ' + layer_name)

    for nn in range(0, n):
        for cc in range(0, c):
            for hh in range(0, h):
                for ww in range(0, w):
                    ss = struct.pack('f', weights[nn, cc, hh, ww])
                    gpu_file.write(ss)

def convert_weights_fc(weights, gpu_file, layer_name):
    n = weights.shape[0]
    c = weights.shape[1]
    if n % 8 != 0:
        print('warning: for GPU models, channel should be divided by 8, check manually' + layer_name)

    for nn in range(0, n):
        for cc in range(0, c):
                ss = struct.pack('f', weights[nn, cc])
                gpu_file.write(ss)

def convert_weights_8bit(weights, gpu_file, layer_name, quantization_param):
    n = weights.shape[0]
    c = weights.shape[1]
    h = weights.shape[2]
    w = weights.shape[3]
    if n % 8 != 0:
        print('for GPU models, channel must be divided by 8: ' + layer_name)

    for nn in range(0, n):
        for cc in range(0, c):
            for hh in range(0, h):
                for ww in range(0, w):
                    weight_8bit = unittest_generate_samples.float_to_fix(weights[nn, cc, hh, ww], quantization_param.bw_params, quantization_param.fl_params, False)
                    ss = struct.pack('b', weight_8bit)
                    gpu_file.write(ss)

def convert_weights_fc_8bit(weights, gpu_file, layer_name, quantization_param):
    n = weights.shape[0]
    c = weights.shape[1]
    if n % 8 != 0:
        print('warning: for GPU models, channel should be divided by 8, check manually' + layer_name)

    for nn in range(0, n):
        for cc in range(0, c):
                weight_8bit = unittest_generate_samples.float_to_fix(weights[nn, cc], quantization_param.bw_params, quantization_param.fl_params, False)
                ss = struct.pack('b', weight_8bit)
                gpu_file.write(ss)

def convert_weights_16bit(weights, gpu_file, layer_name):
    n = weights.shape[0]
    c = weights.shape[1]
    h = weights.shape[2]
    w = weights.shape[3]
    if n % 8 != 0:
        print('for GPU models, channel must be divided by 8: ' + layer_name)

    for nn in range(0, n):
        for cc in range(0, c):
            for hh in range(0, h):
                for ww in range(0, w):
                    #print(np.float16(weights[nn, cc, hh, ww]))
                    tmp_bytes = np.float16(weights[nn, cc, hh, ww]).tobytes()
                    ss0 = struct.pack('c', tmp_bytes[0])
                    ss1 = struct.pack('c', tmp_bytes[1])
                    gpu_file.write(ss0)
                    gpu_file.write(ss1)

def convert_weights_fc_16bit(weights, gpu_file, layer_name):
    n = weights.shape[0]
    c = weights.shape[1]
    if n % 8 != 0:
        print('for GPU models, channel must be divided by 8: ' + layer_name)

    for nn in range(0, n):
        for cc in range(0, c):
                #weight_8bit = unittest_generate_samples.float_to_fix(weights[nn, cc], quantization_param.bw_params, quantization_param.fl_params, False)
                tmp_bytes = np.float16(weights[nn, cc]).tobytes()
                ss0 = struct.pack('c', tmp_bytes[0])
                ss1 = struct.pack('c', tmp_bytes[1])
                gpu_file.write(ss0)
                gpu_file.write(ss1)


def convert_weights_16bit_format(caffe_proto, caffemodel, minidnn_weight_path):
    print("convert minidnn for: ", caffe_proto)
    caffe_weights = caffe.Net(caffe_proto, caffemodel, caffe.TEST)
    net_params = openCaffeSpec(caffe_proto)
    gpu_file = open(minidnn_weight_path,'wb')
    idx = 0
    for layer in caffe_weights.layers:
        if 'Convolution' == layer.type:
            weights = layer.blobs[0].data
            quantization_param = get_layer_params(net_params, caffe_weights._layer_names[idx], 'quantization_param')
            convert_weights_16bit(weights, gpu_file, caffe_weights._layer_names[idx])
            if len(layer.blobs) == 2:
                bias = layer.blobs[1].data
                n = bias.shape[0]
                for nn in range(0, n):
                    tmp_bytes = np.float16(bias[nn]).tobytes()
                    ss0 = struct.pack('c', tmp_bytes[0])
                    ss1 = struct.pack('c', tmp_bytes[1])
                    gpu_file.write(ss0)
                    gpu_file.write(ss1)
        elif 'InnerProduct' == layer.type:
            weights = layer.blobs[0].data
            quantization_param = get_layer_params(net_params, caffe_weights._layer_names[idx], 'quantization_param')
            convert_weights_fc_16bit(weights, gpu_file, caffe_weights._layer_names[idx])
            if len(layer.blobs) == 2:
                bias = layer.blobs[1].data
                n = bias.shape[0]
                for nn in range(0, n):
                    tmp_bytes = np.float16(bias[nn]).tobytes()
                    ss0 = struct.pack('c', tmp_bytes[0])
                    ss1 = struct.pack('c', tmp_bytes[1])
                    gpu_file.write(ss0)
                    gpu_file.write(ss1)
        elif 'Deconvolution' == layer.type:
            assert (len(layer.blobs)==1)  #deconv bias should be zeros
            weights = layer.blobs[0].data
            # c = weights.shape[0]
            # n = weights.shape[1]
            # h = weights.shape[2]
            # w = weights.shape[3]
            #
            # weights_conv = np.zeros(shape=(n,c,h,w), dtype=np.float32)
            #
            # for nn in range(0, n):
            #     for cc in range(0, c):
            #         for hh in range(0, h):
            #             for ww in range(0, w):
            #                 weights_conv[cc,nn,hh,ww] = weights[nn,cc,hh,ww]

            convert_weights_16bit(weights, gpu_file, caffe_weights._layer_names[idx])
            if len(layer.blobs) == 2:
                raise Exception ('Not implemented bias for deconv')
        else:
            None

        #print('processed layer:', caffe_weights._layer_names[idx])
        idx += 1
    gpu_file.close()
    print('check mean value first')


def convert_weights_32f_format(caffe_proto, caffemodel, minidnn_weight_path):
    print("convert minidnn for: ", caffe_proto)
    caffe_weights = caffe.Net(caffe_proto, caffemodel, caffe.TEST)
    gpu_file = open(minidnn_weight_path,'wb')
    idx = 0
    for layer in caffe_weights.layers:
        if 'Convolution' == layer.type:
            weights = layer.blobs[0].data
            convert_weights(weights, gpu_file, caffe_weights._layer_names[idx])
            if len(layer.blobs) == 2:
                bias = layer.blobs[1].data
                n = bias.shape[0]
                for nn in range(0, n):
                    ss = struct.pack('f', bias[nn])
                    gpu_file.write(ss)
        elif 'InnerProduct' == layer.type:
            weights = layer.blobs[0].data
            convert_weights_fc(weights, gpu_file, caffe_weights._layer_names[idx])
            if len(layer.blobs) == 2:
                bias = layer.blobs[1].data
                n = bias.shape[0]
                for nn in range(0, n):
                    ss = struct.pack('f', bias[nn])
                    gpu_file.write(ss)
        elif 'Deconvolution' == layer.type:
            assert (len(layer.blobs) == 1)  # deconv bias should be zeros
            weights = layer.blobs[0].data
            # c = weights.shape[0]
            # n = weights.shape[1]
            # h = weights.shape[2]
            # w = weights.shape[3]
            #
            # weights_conv = np.zeros(shape=(n, c, h, w), dtype=np.float32)
            #
            # for nn in range(0, n):
            #     for cc in range(0, c):
            #         for hh in range(0, h):
            #             for ww in range(0, w):
            #                 weights_conv[nn, cc, hh, ww] = weights[cc, nn, hh, ww]

            convert_weights(weights, gpu_file, caffe_weights._layer_names[idx])
            if len(layer.blobs) == 2:
                raise Exception('Not implemented bias for deconv')
        else:
            None

        #print('processed layer:', caffe_weights._layer_names[idx])
        idx += 1
    gpu_file.close()
    print('check mean value first')



def convert_weights_8bit_format(caffe_proto, caffemodel, minidnn_weight_path):
    print("convert minidnn for: ", caffe_proto)
    caffe_weights = caffe.Net(caffe_proto, caffemodel, caffe.TEST)
    net_params = openCaffeSpec(caffe_proto)
    gpu_file = open(minidnn_weight_path,'wb')
    idx = 0
    for layer in caffe_weights.layers:
        if 'ConvolutionRistretto' == layer.type:
            weights = layer.blobs[0].data
            quantization_param = get_layer_params(net_params, caffe_weights._layer_names[idx], 'quantization_param')
            convert_weights_8bit(weights, gpu_file, caffe_weights._layer_names[idx], quantization_param)
            if len(layer.blobs) == 2:
                bias = layer.blobs[1].data
                n = bias.shape[0]
                for nn in range(0, n):
                    bias_8bit = unittest_generate_samples.float_to_fix(
                        bias[nn],quantization_param.bw_params, quantization_param.fl_bias, False)
                    ss = struct.pack('b', bias_8bit)
                    gpu_file.write(ss)
        elif 'InnerProductRistretto' == layer.type:
            weights = layer.blobs[0].data
            quantization_param = get_layer_params(net_params, caffe_weights._layer_names[idx], 'quantization_param')
            convert_weights_fc_8bit(weights, gpu_file, caffe_weights._layer_names[idx], quantization_param)
            if len(layer.blobs) == 2:
                bias = layer.blobs[1].data
                n = bias.shape[0]
                for nn in range(0, n):
                    bias_8bit = unittest_generate_samples.float_to_fix(
                        bias[nn],quantization_param.bw_params, quantization_param.fl_bias, False)
                    ss = struct.pack('b', bias_8bit)
                    gpu_file.write(ss)
        elif 'DeconvolutionRistretto' == layer.type:
            assert (len(layer.blobs) == 1)  # deconv bias should be zeros
            weights = layer.blobs[0].data
            # c = weights.shape[0]
            # n = weights.shape[1]
            # h = weights.shape[2]
            # w = weights.shape[3]
            #
            # weights_conv = np.zeros(shape=(n, c, h, w), dtype=np.float32)
            #
            # for nn in range(0, n):
            #     for cc in range(0, c):
            #         for hh in range(0, h):
            #             for ww in range(0, w):
            #                 weights_conv[nn, cc, hh, ww] = weights[cc, nn, hh, ww]
            quantization_param = get_layer_params(net_params, caffe_weights._layer_names[idx], 'quantization_param')
            convert_weights_8bit(weights, gpu_file, caffe_weights._layer_names[idx], quantization_param)
            if len(layer.blobs) == 2:
                raise Exception('Not implemented bias for deconv')
        else:
            None

        #print('weights converted for layer:', caffe_weights._layer_names[idx])
        idx += 1
    gpu_file.close()
    print('check mean value first')

def minidnn_auto_covert_8bit(prototxt_path, caffe_model, mean_val, scale, output_layers, platform):
    json_path = prototxt_path.replace('.prototxt', '_8bit.json')
    weight_path = prototxt_path.replace('.prototxt', '_8bit.txt')
    net_spec = openCaffeSpec(prototxt_path)
    total_complexity, total_model_size, complexity_dict, model_size_dict, shape_dict = \
        infer_complexity.infer_complexity(prototxt_path, output_layers)
    serialize_json_8bit.record_json(net_spec, shape_dict, mean_val, scale, platform, json_path)
    convert_weights_8bit_format(prototxt_path, caffe_model, weight_path)

def minidnn_auto_covert_fp16(prototxt_path, caffe_model, mean_val, scale, output_layers):
    json_path_cpu = prototxt_path.replace('.prototxt', '_cpu_fp16.json')
    #json_path_mali = prototxt_path.replace('.prototxt', '_mali_fp16.json')
    weight_path = prototxt_path.replace('.prototxt', '_fp16.txt')
    net_spec = openCaffeSpec(prototxt_path)
    total_complexity, total_model_size, complexity_dict, model_size_dict, shape_dict = \
        infer_complexity.infer_complexity(prototxt_path, output_layers)
    serialize_json_fp16.record_json(net_spec, shape_dict, mean_val, scale, "cpu", json_path_cpu)
    #serialize_json_fp16.record_json(net_spec, shape_dict, mean_val, scale, "mali", json_path_mali)
    convert_weights_16bit_format(prototxt_path, caffe_model, weight_path)

if __name__ == '__main__':

    prototxt_path = sys.argv[1]#'/data1/wzheng/projects/model_zoo_arm_8bit/dms/face_landmark/models_20191205/deploy_detnet_nobn.prototxt'
    caffe_model = sys.argv[2]#'/data1/wzheng/projects/model_zoo_arm_8bit/dms/face_landmark/models_20191205/deploy_detnet_nobn_20191205.caffemodel'
    mean_val = 0.0
    scale = 1/256.0
    platform = 'cpu'
    output_layers = ['fc']

    minidnn_auto_covert_fp16(prototxt_path, caffe_model, mean_val, scale, output_layers)