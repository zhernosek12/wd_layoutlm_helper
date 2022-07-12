from PIL import Image, ImageDraw, ImageFont

def normalize_box(box, width, height):
     return [
         int(1000 * (box[0] / width)),
         int(1000 * (box[1] / height)),
         int(1000 * (box[2] / width)),
         int(1000 * (box[3] / height)),
     ]

def apply_ocr(json_file, width, height):
        # get the image
        all = []
        example = {}
       
        # get json
        with open(json_file, encoding="utf8")as f:
            json_string = f.read()
            responce = json.loads(json_string)

        words = []
        actual_boxes = []
        
        assert(len(responce["results"]) == 1)
        assert(len(responce["results"][0]['results'][0]) == 1)

        for page in responce["results"][0]['results'][0]['textDetection']['pages']:
            #width = page['width']
            #height = page['height']
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
            boxes.append(normalize_box(box, width, height))
        
        # add as extra columns 
        assert len(words) == len(boxes)

        example['words'] = words
        example['boxes'] = boxes
        example['actual_boxes'] = actual_boxes
        
        return example


def my_processor(paths, return_tensors=""):
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
        example = apply_ocr(json_file, width, height)

        words = example['words']
        boxes = example['boxes']
        actual_boxes = example['actual_boxes']

        encoded_inputs = processor(image, words, boxes=boxes,
                                   padding="max_length", truncation=True, 
                                   return_tensors=return_tensors)

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

def encode_training_example(examples):
    '''
    images = [Image.open(path).convert("RGB") for path in examples['image_path']]
    encoded_inputs = processor(images, padding="max_length", truncation=True)
    encoded_inputs["labels"] = [label2idx[label] for label in examples["label"]]
    # 
    ''' 
    
    images = [Image.open(path).convert("RGB") for path in examples['image_path']]
    encoded_inputs = my_processor(examples['image_path'])
    encoded_inputs["labels"] = [label2idx[label] for label in examples["label"]]
    
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
