import argparse
import matplotlib.pyplot as plot
import numpy as np

from chainer import serializers

from chainercv.datasets.pascal_voc import voc_utils
from chainercv.links import SSD300
from chainercv import utils
from chainercv.visualizations import vis_bbox


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('image')
    args = parser.parse_args()

    model = SSD300(n_class=20)
    serializers.load_npz(args.model, model)

    img = utils.read_image(args.image, color=True)
    bboxes, labels, scores = model.predict(img[np.newaxis])
    bbox, label, score = bboxes[0], labels[0], scores[0]

    vis_bbox(
        img, bbox, label, score, label_names=voc_utils.pascal_voc_labels)
    plot.show()


if __name__ == '__main__':
    main()
