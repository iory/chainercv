import numpy as np
import os
import warnings
import json

from chainercv.chainer_experimental.datasets.sliceable import GetterDataset
from chainercv.datasets.coco import coco_utils
from chainercv.utils import read_image


class CocoPersonKeypointsDataset(GetterDataset):

    def __init__(self, data_dir='auto', split='train', year='2017'):
        super(CocoPersonKeypointsDataset, self).__init__()

        if data_dir == 'auto' and year in ['2017']:
            data_dir = coco_utils.get_coco(year)

        # if split not in ['train', 'trainval', 'val']:
        #     if not (split == 'test' and year == '2007'):
        #         warnings.warn(
        #             'please pick split from \'train\', \'trainval\', \'val\''
        #             'for 2012 dataset. For 2007 dataset, you can pick \'test\''
        #             ' in addition to the above mentioned splits.'
        #         )
        self.id_list_file = json.load(open(
            os.path.join(data_dir,
                         'annotations/person_keypoints_{0}{1}.json'.
                         format(split, year))))

        self.filenames = [id_['file_name'] for id_ in self.id_list_file['images']]

        self.data_dir = os.path.join(data_dir, '{}{}'.format(split, year))
        # self.use_difficult = use_difficult

        self.add_getter('img', self._get_image)
        # self.add_getter(('bbox', 'label', 'difficult'), self._get_annotations)

        # if not return_difficult:
        #     self.keys = ('img', 'bbox', 'label')

    def __len__(self):
        return len(self.filenames)

    def _get_image(self, i):
        filename = self.filenames[i]
        img_path = os.path.join(self.data_dir, filename)
        img = read_image(img_path, color=True)
        return img

    def _get_ignore_mask(self, i):
        id_ = self.ids[i]

    def _get_annotations(self, i):
        id_ = self.ids[i]
        anno = ET.parse(
            os.path.join(self.data_dir, 'Annotations', id_ + '.xml'))
        bbox = []
        label = []
        difficult = []
        for obj in anno.findall('object'):
            # when in not using difficult split, and the object is
            # difficult, skipt it.
            if not self.use_difficult and int(obj.find('difficult').text) == 1:
                continue

            difficult.append(int(obj.find('difficult').text))
            bndbox_anno = obj.find('bndbox')
            # subtract 1 to make pixel indexes 0-based
            bbox.append([
                int(bndbox_anno.find(tag).text) - 1
                for tag in ('ymin', 'xmin', 'ymax', 'xmax')])
            name = obj.find('name').text.lower().strip()
            label.append(voc_utils.voc_bbox_label_names.index(name))
        bbox = np.stack(bbox).astype(np.float32)
        label = np.stack(label).astype(np.int32)
        # When `use_difficult==False`, all elements in `difficult` are False.
        difficult = np.array(difficult, dtype=np.bool)
        return bbox, label, difficult
