import argparse



try:
    from main import ImageSearch
except:
    from source.main import ImageSearch
if __name__ == '__main__':
    # Create parser and add arguments
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-d', '--dataset', help='Name of dataset')

    # Create arguments
    args = argParser.parse_args()
    models = ['VisionTransformer','BEiT','MobileViTV2','Bit','EfficientFormer','MobileNetV2','ResNet','EfficientNet']
    similarity_function = ['cosine','TS_SS','euclidean']
    for i in models:
        IS = ImageSearch(args.dataset, i)
        IS.indexing()
        for j in similarity_function:
            result = ImageSearch(args.dataset, i,j)
            result.evaluating()