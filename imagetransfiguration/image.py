from loguru import logger
import numpy as np
from sklearn.cluster import KMeans


class Image:

    @staticmethod
    @logger.catch
    def find_centroids(image, n, boxes=None) -> [list, list]:
        """

        :param image: np.array with binary image dtype uint8
        :param boxes: list of lists bounding boxes with coordinates x1, y1, x2, y2
        :param n: int number of centroids
        :return: list with x centroid, list with y centroids
        """
        assert n is not None and n > 0
        image = cv2.medianBlur(image, 3)
        centre_x, centre_y = [], []
        if boxes is None:
            boxes = [[0, 0, image.shape[1] - 1, image.shape[0] - 1]]
        for box in boxes:
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            y1_crop = max(0, y1)
            y2_crop = min(image.shape[0] - 1, y2)
            x1_crop = max(0, x1)
            x2_crop = min(image.shape[1] - 1, x2)
            image = image[y1_crop:y2_crop, x1_crop:x2_crop]
            x, y = np.where(image == 255)
            points = np.array([x, y]).T
            if points.shape[0] < n:
                return binary_blobs, centre_x, centre_y
            k_means = KMeans(n_clusters=n)
            k_means.fit(points)
            centroid = k_means.cluster_centers_
            centroid_x = centroid[:, 1] + x1
            centroid_y = centroid[:, 0] + y1
            centroid_x, centroid_y = np.array(centroid_x, dtype='uint16'), np.array(centroid_y, dtype='uint16')
            centre_x.append(centroid_x)
            centre_y.append(centroid_y)
        return image, centre_x, centre_y


if __name__ == "__main__":
    import cv2
    from generator import DummyCircles

    circles = DummyCircles()(n_circles=5, max_radius=20, width=500, height=500)
    binary_blobs = np.zeros((500, 500), dtype='uint8')
    for rr, cc in circles:
        binary_blobs[rr, cc] = 255
    binary_blobs, central_x, central_y = Image.find_centroids(binary_blobs, 5)
    binary_blobs = cv2.cvtColor(binary_blobs, cv2.COLOR_GRAY2BGR)
    for x, y in zip(*central_x, *central_y):
        binary_blobs = cv2.circle(binary_blobs, (int(x), int(y)), radius=3, color=(0, 255, 0), thickness=-1)

    cv2.imshow("binary_blobs", binary_blobs)
    cv2.waitKey(0)
