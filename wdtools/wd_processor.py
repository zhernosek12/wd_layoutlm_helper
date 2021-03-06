import os
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from datasets import Dataset, Features, Sequence, ClassLabel, Value, Array2D, Array3D
from transformers import LayoutLMv2FeatureExtractor, LayoutLMv2Tokenizer, LayoutLMv2Processor, LayoutLMv2ForSequenceClassification, AdamW

class WdDataset:

    def __init__(self):
        self.label2idx = {}
        self.processor = None
        self.batch_size = 4
        self.device = None

    def normalize_box(self, box, width, height):
        return [
            int(1000 * (box[0] / width)),
            int(1000 * (box[1] / height)),
            int(1000 * (box[2] / width)),
            int(1000 * (box[3] / height)),
        ]

    def apply_ocr(self, json_file, read=True):
            # get the image
            all = []
            example = {}
          
            # get json
            if read == True:
                with open(json_file, encoding="utf8") as f:
                    json_string = f.read()
                    responce = json.loads(json_string)
            else:
                responce = json_file

            words = []
            actual_boxes = []
            
            assert(len(responce["results"]) == 1)
            assert(len(responce["results"][0]['results'][0]) == 1)

            for page in responce["results"][0]['results'][0]['textDetection']['pages']:
                width = int(page['width'])
                height = int(page['height'])
                for block in page["blocks"]:
                    for line in block["lines"]:
                        for word in line["words"]:
                            vertices = word["boundingBox"]["vertices"]

                            x_all = [int(v["x"]) for v in vertices if "x" in v]
                            y_all = [int(v["y"]) for v in vertices if "y" in v]

                            if len(x_all) > 0:
                                minx = min(x_all)
                                maxx = max(x_all)
                            else:
                                minx = 0
                                maxx = 0

                            if len(y_all) > 0:
                                miny = min(y_all)
                                maxy = max(y_all)
                            else:
                                miny = 0
                                maxy = 0

                            text = word["text"]
                            x1 = max(0, int(minx))
                            y1 = max(0, int(miny))
                            x2 = max(x1 + 1, int(maxx))
                            y2 = max(y1 + 1, int(maxy))

                            words.append(text)
                            actual_boxes.append([x1, y1, x2, y2])


            
            # normalize the bounding boxes
            boxes = []
            for box in actual_boxes:
                boxes.append(self.normalize_box(box, width, height))
            
            # add as extra columns 
            assert len(words) == len(boxes)

            example['words'] = words
            example['boxes'] = boxes
            example['actual_boxes'] = actual_boxes
            
            return example


    def my_processor(self, paths):
        examples = {
            "image": [],
            "input_ids": [],
            "bbox": [],
            "token_type_ids": [],
            "attention_mask": [],
        }
        for image_path in paths:

            image = Image.open(image_path)
            width, height = image.size
            
            json_file = image_path.replace(".png", ".json")
            example = self.apply_ocr(json_file)

            words = example['words']
            boxes = example['boxes']
            actual_boxes = example['actual_boxes']

            encoded_inputs = self.processor(image, words, boxes=boxes,
                                      padding="max_length", truncation=True)

            examples["image"].append(encoded_inputs["image"][0])
            examples["input_ids"].append(encoded_inputs["input_ids"])
            examples["bbox"].append(encoded_inputs["bbox"])
            examples["token_type_ids"].append(encoded_inputs["token_type_ids"])
            examples["attention_mask"].append(encoded_inputs["attention_mask"])

            #examples["image"].append(np.array(image))
            #examples["input_ids"].append(input_ids)
            #examples["bbox"].append(bbox)
            #examples["token_type_ids"].append(token_type_ids)
            #examples["attention_mask"].append(attention_mask)

        return examples

    def encode_training_example(self, examples):
        '''
        images = [Image.open(path).convert("RGB") for path in examples['image_path']]
        encoded_inputs = processor(images, padding="max_length", truncation=True)
        encoded_inputs["labels"] = [label2idx[label] for label in examples["label"]]
        # 
        ''' 
        
        images = [Image.open(path).convert("RGB") for path in examples['image_path']]
        encoded_inputs = self.my_processor(examples['image_path'])
        encoded_inputs["labels"] = [self.label2idx[label] for label in examples["label"]]
        
        ''' 
        print('input', examples.keys())
        print('----------------------------')
        print(encoded_inputs.keys())
        for k in encoded_inputs.keys():
            print('-', k, type(encoded_inputs[k]), len(encoded_inputs[k]))
            print('--', type(encoded_inputs[k][0]), encoded_inputs[k][0])
        print('--------------------------------------------')
        ''' 

        return encoded_inputs

    def training_dataloader_from_df(self, data):
        dataset = Dataset.from_pandas(data)
        training_features = Features({
            'image': Array3D(dtype="int64", shape=(3, 224, 224)),
            'input_ids': Sequence(feature=Value(dtype='int64')),
            'attention_mask': Sequence(Value(dtype='int64')),
            'token_type_ids': Sequence(Value(dtype='int64')),
            'bbox': Array2D(dtype="int64", shape=(512, 4)),
            'labels': ClassLabel(num_classes=len(self.label2idx), names=list(self.label2idx.keys())),
        })
        encoded_dataset = dataset.map(
            self.encode_training_example, remove_columns=dataset.column_names, features=training_features, 
            batched=True, batch_size=2
        )
        encoded_dataset.set_format(type='torch', device=self.device)
        dataloader = torch.utils.data.DataLoader(encoded_dataset, batch_size=self.batch_size, shuffle=True)
        batch = next(iter(dataloader))
        return dataloader
