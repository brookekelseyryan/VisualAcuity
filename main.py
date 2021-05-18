try:
    import os, sys

    import helloworld

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    assert int(sys.argv[1]) in [0, 1, 2, 3]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(sys.argv[1])

    print("Hello world!!!")

    helloworld.main()
except BaseException as error:
    print('An exception occurred: {}'.format(error))

