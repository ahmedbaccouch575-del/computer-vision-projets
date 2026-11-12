from langchain.tools import BaseTool
from transformers import BlipProcessor, BlipForConditionalGeneration, DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import torch


class ImageCaptionTool(BaseTool):
    name = "Image captioner"
    description = (
        "Use this tool when given the path to an image that you would like to be described. "
        "It will return a simple caption describing the image."
    )

    def _run(self, img_path):
        image = Image.open(img_path).convert("RGB")

        model_name = "Salesforce/blip-image-captioning-large"
        device = "cuda" if torch.cuda.is_available() else "cpu"

        processor = BlipProcessor.from_pretrained(model_name)
        model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)

        inputs = processor(image, return_tensors="pt").to(device)
        output = model.generate(**inputs, max_new_tokens=20)

        return processor.decode(output[0], skip_special_tokens=True)

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")


class ObjectDetectionTool(BaseTool):
    name = "Object detector"
    description = (
        "Use this tool when given the path to an image that you would like to detect objects. "
        "It will return a list of all detected objects. Each element in the list is in the format: "
        "[x1, y1, x2, y2] class_name confidence_score."
    )

    def _run(self, img_path):
        image = Image.open(img_path).convert("RGB")

        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)

        # convert outputs to COCO API format, keep detections with score > 0.9
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=0.9
        )[0]

        detections = ""
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [int(i) for i in box.tolist()]
            detections += "[{}, {}, {}, {}] ".format(*box)
            detections += "{} {}\n".format(model.config.id2label[int(label)], float(score))

        return detections

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")