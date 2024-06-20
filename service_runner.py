from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import dtlpy as dl
import numpy as np
import logging
import shutil
import time
import cv2
import os

logger = logging.getLogger('quality-estimator')


class ServiceRunner(dl.BaseServiceRunner):

    @staticmethod
    def calculate_darkness_score(img):
        """
        Calculates darkness score of an image using Laplacian variance.
        Computes the average (mean) value of all the pixels in the image. Higher score represents brighter image.

        :param img: numpy array
        :return: darkness score
        """

        return np.mean(img) / 255.0  # Higher score is brighter (less dark)

    @staticmethod
    def calculate_blurriness_score(img):
        """
        Calculates blurriness score of an image using Laplacian variance.
        Higher Laplacian variance indicates a sharper image. This is because a sharp image has more edges and details,
        which results in higher variance when the Laplacian operator is applied.

        :param img: numpy array
        :return: blurriness score
        """
        laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
        scaled_score = 1 / (1 + np.exp(-laplacian_var / 100.0))  # Sigmoid to map score to scale (0.5-1.0)
        slope = 2  # Calculated as (target_max - target_min) / (original_max - original_min)
        offset = -1  # Calculated as target_min - (original_min * slope)
        scaled_score = slope * scaled_score + offset

        return scaled_score

    def calculate_quality_scores(self, item: dl.Item, pbar: tqdm = None):
        try:
            # Reading item
            local_path = os.path.join(os.getcwd(), 'datasets', item.dataset.id)
            image_path = os.path.join(local_path, 'items', item.filename[1:])
            # If executed from trigger
            if not os.path.isfile(image_path):
                img = item.download(save_locally=False, to_array=True)
            else:
                img = cv2.imread(filename=image_path)

            # Resize by finding a reshape factor
            max_shape = max(img.shape[:2])
            if max_shape > 1024:
                resize_factor = 1024 / max_shape

                # Compute the new dimensions
                new_width = int(img.shape[1] * resize_factor)
                new_height = int(img.shape[0] * resize_factor)
                new_size = (new_width, new_height)  # opposite from image.shape, width first
                img = cv2.resize(img, new_size)

            # Convert to gray scale
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Get scores
            blurriness_score = self.calculate_blurriness_score(img=img)
            darkness_score = self.calculate_darkness_score(img=img)

            # Update item's metadata
            if "user" in item.metadata:
                if 'quality_scores' not in item.metadata["user"]:
                    item.metadata["user"]['quality_scores'] = {}
                item.metadata["user"]['quality_scores']['blurriness_score'] = float(blurriness_score)
                item.metadata["user"]['quality_scores']['darkness_score'] = float(darkness_score)
            else:
                item.metadata["user"] = {
                    'quality_scores': {
                        'blurriness_score': float(blurriness_score),
                        'darkness_score': float(darkness_score)
                    }
                }

            item.update()
        except Exception as e:
            logger.warning(f"Error occurred while handling: {e}. Skipped item: {item.id}")

        finally:
            if pbar is not None:
                pbar.update()

        return item

    def dataset_scores_generator(self, dataset: dl.Dataset):
        local_path = os.path.join(os.getcwd(), 'datasets', dataset.id)
        os.makedirs(local_path, exist_ok=True)

        # Default filter
        filters = dl.Filters()
        filters.add(field='metadata.system.mimetype', values='*image*')

        # Score is been calculated to an item only once
        json = dataset.schema.get()
        items_schema_keys = json.get("items").get("keys").keys()
        if (('metadata.user.quality_scores.blurriness_score' in items_schema_keys) and
                ('metadata.user.quality_scores.darkness_score' in items_schema_keys)):  # Both scores are required
            filters.add(field='metadata.user.quality_scores.blurriness_score',
                        values=False,
                        operator=dl.FILTERS_OPERATIONS_EXISTS)
            filters.add(field='metadata.user.quality_scores.darkness_score',
                        values=False,
                        operator=dl.FILTERS_OPERATIONS_EXISTS,
                        method=dl.FILTERS_METHOD_AND)

        dataset.download(local_path=local_path, filters=filters)

        logger.info("Dataset has been downloaded locally")
        pages = dataset.items.list(filters=filters)

        futures = list()
        tic = time.time()
        with ThreadPoolExecutor(max_workers=16) as executor:
            with tqdm(total=pages.items_count, desc='Processing') as pbar:
                for page in pages:
                    for item in page:
                        future = executor.submit(self.calculate_quality_scores,
                                                 **{'item': item, 'pbar': pbar})
                        futures.append(future)
                        for future in as_completed(futures):
                            future.result()
                logger.info(
                    f'Finished calculating quality scores for {pages.items_count} items. Calculation time:  {time.time() - tic}')

        # Remove local paths
        if os.path.exists(local_path):
            shutil.rmtree(local_path)


if __name__ == '__main__':
    dl.setenv('rc')
    # project = dl.projects.get(project_id='2bb16c5f-081f-4afb-91e0-78699c1b3476')  # Embeddings Demo (CLIP)
    # project = dl.projects.get(project_name='text-project')  # Embeddings Demo (CLIP)
    project = dl.projects.get(project_name='roni-tests')

    # dataset = project.datasets.get(dataset_name='Training Set')
    # dataset = project.datasets.get(dataset_name='roni-test')
    dataset = project.datasets.get(dataset_name='ImageNet')

    # dataset = project.datasets.get(dataset_name='test-quality')
    s = ServiceRunner()
    s.dataset_scores_generator(dataset=dataset)
    # item = dl.items.get(item_id='6628f6477ad6f3dbb091a470')
    # s.calculate_quality_scores(item)
