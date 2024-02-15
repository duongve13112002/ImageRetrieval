import time
import cv2 as cv
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from pickle import dump, load
import os

try:
    import feature_extractor as fe
    from TS_SS_similarity import TS_SS
except:
    import source.feature_extractor as fe
    from source.TS_SS_similarity import TS_SS


def euclidean(a, b):
    return np.linalg.norm(a - b)
def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

class ImageSearch:
    def __init__(self, dataset_name='oxbuild', method='VisionTransformer',method_calculate_vectors = 'cosine') -> None:
        if method == 'VisionTransformer':
            self.feature_extractor = fe.VisionTransformer()
        elif method == 'BEiT':
            self.feature_extractor = fe.BEiT()
        elif method == 'MobileViTV2':
            self.feature_extractor = fe.MobileViTV2()
        elif method == 'Bit':
            self.feature_extractor = fe.Bit()
        elif method == 'EfficientFormer':
            self.feature_extractor = fe.EfficientFormer()
        elif method == 'MobileNetV2':
            self.feature_extractor = fe.MobileNetV2()
        elif method == 'ResNet':
            self.feature_extractor = fe.ResNet()
        elif method == 'EfficientNet':
            self.feature_extractor = fe.EfficientNet()

        self.method = method

        if method_calculate_vectors == 'cosine':
            self.method_calculate_vectors = cosine
            self.name_function = 'cosine'
            self.compare_different = 'greater better'
        elif method_calculate_vectors == 'TS_SS':
            self.method_calculate_vectors = TS_SS()
            self.name_function = 'TS_SS'
            self.compare_different = 'smaller better'
        else:
            self.method_calculate_vectors = euclidean
            self.name_function = 'euclidean'
            self.compare_different = 'smaller better'

        self.dataset_name = dataset_name
        self.subdirectories = []
        self.base_path = Path('./static/datasets')
        if self.dataset_name == 'All Database':
            files_and_folders = [item for item in os.listdir(self.base_path) if not item.startswith('.')]
            subdirectories = [item for item in files_and_folders if os.path.isdir(os.path.join(self.base_path, item))]
            self.subdirectories = subdirectories
        self.dataset_folder_path = Path('./static/datasets') / dataset_name
        if not self.dataset_folder_path.exists():
            self.dataset_folder_path.mkdir()


        self.images_folder_path = self.dataset_folder_path / 'images'
        if not self.images_folder_path.exists():
            self.images_folder_path.mkdir()


        self.feature_map_folder_path = self.dataset_folder_path / 'feature map'
        if not self.feature_map_folder_path.exists():
            self.feature_map_folder_path.mkdir()


        self.methods_folder_path = self.feature_map_folder_path / self.method
        if not self.methods_folder_path.exists():
            self.methods_folder_path.mkdir()


        self.groundtruth_folder_path = self.dataset_folder_path / 'groundtruth'
        if not self.groundtruth_folder_path.exists():
            self.groundtruth_folder_path.mkdir()

    def indexing(self) -> None:
        features_file_path = self.methods_folder_path / 'features.pkl'
        image_paths_file_path = self.methods_folder_path / 'images.pkl'

        if not features_file_path.exists() or not image_paths_file_path.exists():
            features, image_paths = [], []

            if self.dataset_name == 'All Database' and len(self.subdirectories) > 0:
                for database in self.subdirectories:
                    images_folder = self.base_path / database / 'images'
                    for image_path in tqdm(sorted(images_folder.glob('*.jpg'))):
                        try:
                            image = Image.open(image_path)

                            feature = self.feature_extractor.extract(image)

                            features.append(feature)
                            image_paths.append(image_path)
                        except:
                            continue
            else:
                for image_path in tqdm(sorted(self.images_folder_path.glob('*.jpg'))):
                    try:
                        image = Image.open(image_path)

                        feature = self.feature_extractor.extract(image)

                        features.append(feature)
                        image_paths.append(image_path)
                    except:
                        continue

            features_file = open(features_file_path, 'wb')
            image_paths_file = open(image_paths_file_path, 'wb')

            dump(features, features_file)
            dump(image_paths, image_paths_file)

    def retrieve_image(self, query_image, K=16) -> tuple:
        start = time.time()

        features_file_path = self.methods_folder_path / 'features.pkl'
        image_paths_file_path = self.methods_folder_path / 'images.pkl'

        features_file = open(features_file_path, 'rb')
        image_paths_file = open(image_paths_file_path, 'rb')

        features = load(features_file)
        image_paths = load(image_paths_file)

        if isinstance(query_image, str):  
            query_image = Image.open(query_image)  

        query_feature = self.feature_extractor.extract(query_image)  

        scores = []

        for feature in features:
            score = self.method_calculate_vectors(query_feature, feature)
            scores.append(score) 

        if self.compare_different == 'greater better':
            topk_ids = np.argsort(scores)[::-1][:K]
        else:
            topk_ids = np.argsort(scores)[::][:K]
        

        ranked = [(image_paths[id], scores[id]) for id in topk_ids]

        end = time.time()

        return ranked, end - start

    def AP_MRR(self, predict, groundtruth):
        # Create precision, recall,mrr list
        p = []
        r = []
        mrr = []
        correct = 0  # Number of correct images

        # Browse each image in predict list
        for id, (image, score) in enumerate(predict):
            # If image in groundtruth
            if image.stem in groundtruth:
                correct += 1  # Increase number of correct images
                p.append(correct / (id + 1))  # Add precision at this position
                mrr.append(1/(id+1)) #mRR
                #interpolated AP
                r.append(correct / len(groundtruth))  # Add recall at this position

        # interpolated AP
        trec = []  # Calculate precision at 11 point of TREC
        for R in range(11):  # Browse 11 point of recall from 0 to 1 with step 0.1
            pm = []  # Create precision list to find max precision
            for id, pr in enumerate(p):  # Browse each precision above
                if r[id] >= R / 10:  # If corresponding recall is greater than or equal to this point
                    pm.append(pr)  # Add precision to precision list to find max
            trec.append(max(pm) if len(pm) else 0)  # Add max precision at this point to trec

        return np.mean(np.array(p)) if len(p) else 0,np.mean(np.array(trec)) if len(trec) else 0,mrr[0] if len(mrr) else 0  # Return non - interpolated AP,interpolated AP,mrr

    def evaluating(self):
        results_folder_path = self.dataset_folder_path / 'results'
        if not results_folder_path.exists():
            results_folder_path.mkdir()

        result_file_path = results_folder_path / f'{self.dataset_folder_path.stem}_{self.method}_{self.name_function}_evaluation.txt'
        if result_file_path.exists():
            result_file_path.unlink()

        result_file = open(result_file_path, 'a')

        result_file.write('-' * 20 + 'START EVALUATING' + '-' * 20 + '\n\n')

        start = time.time()
        
        if self.dataset_name == 'All Database' and len(self.subdirectories) > 0:
            queries_file_lst = []
            for database in self.subdirectories:
                groundtruth_folder = self.base_path / database / 'groundtruth'
                queries_file_lst.append(sorted(groundtruth_folder.glob('*_query.txt')))
            queries_file = [item for sublist in queries_file_lst for item in sublist]
        else:
            queries_file = sorted(self.groundtruth_folder_path.glob('*_query.txt'))

        nAPs = []
        iAPs = []
        mRRs =[]

        for id, query_file in enumerate(queries_file):
            groundtruth = []

            with open(str(query_file).replace('query', 'good'), 'r') as groundtruth_file:
                groundtruth.extend([line.strip() for line in groundtruth_file.readlines()])

            with open(str(query_file).replace('query', 'ok'), 'r') as groundtruth_file:
                groundtruth.extend([line.strip() for line in groundtruth_file.readlines()])

            G = len(groundtruth)

            with open(query_file, 'r') as query:
                content = query.readline().strip().split()

                
                if 'oxc1_' in content[0]:
                    replace_str = 'oxc1_'
                else:
                    replace_str = ''

                # Get image path and coordinates of bounding box
                image_name = content[0].replace(replace_str, '') + '.jpg'
                if self.dataset_name == 'All Database' and len(self.subdirectories) > 0:
                    for database in self.subdirectories:
                        images_folder = self.base_path / database / 'images'
                        files_and_folders = [item for item in os.listdir(images_folder) if not item.startswith('.')]
                        if image_name in files_and_folders:
                            image_path=images_folder / image_name
                            break
                else:
                    image_path = self.images_folder_path / image_name
                
                bounding_box = tuple(float(coor) for coor in content[1:])

                query_image = Image.open(image_path)
                query_image = query_image.crop(bounding_box)

                rel_imgs, query_time = self.retrieve_image(query_image, G)

                nAP,iAP,mrr = self.AP_MRR(rel_imgs, groundtruth)

                nAPs.append(nAP)
                iAPs.append(iAP)
                mRRs.append(mrr)

            result_file.write(f'+ Query {(id + 1):2d}: {Path(query_file).stem}.txt\n')

            result_file.write(' ' * 12 + f'Non - Interpolated Average Precision = {nAP:.5f}\n')
            result_file.write(' ' * 12 + f'Interpolated Average Precision = {iAP:.5f}\n')

            result_file.write(' ' * 12 + f'Query Time = {query_time:2.2f}s\n')

        end = time.time()

        result_file.write('\n' + '-' * 19 + 'FINISH EVALUATING' + '-' * 20 + '\n\n')

        nMAP = np.mean(np.array(nAPs))
        iMAP = np.mean(np.array(iAPs))
        mRR = np.mean(np.array(mRRs))

        result_file.write(f'Total number of queries = {len(queries_file)}\n')

        result_file.write(f'Non - Interpolated Mean Average Precision = {nMAP:.5f}\n')
        result_file.write(f'Interpolated Mean Average Precision = {iMAP:.5f}\n')
        result_file.write(f'Mean Reciprocal Rank = {mRR:.5f}\n')

        result_file.write(f'Evaluating Time = {(end - start):2.2f}s')

        result_file.close()