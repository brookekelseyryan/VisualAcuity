# import os, sys
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# assert int(sys.argv[1]) in [0,1,2,3]
# os.environ["CUDA_VISIBLE_DEVICES"]=str(sys.argv[1])

def main():
    print("hello wurld")
    import os, sys
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())
main()