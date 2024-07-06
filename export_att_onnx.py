import torch
import argparse
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from torchvision import transforms as T
from typing import Union

from easyface.attributes.models import *
from easyface.utils.visualize import show_image
from easyface.utils.io import WebcamStream, VideoReader, VideoWriter, FPS
from detect_align import FaceDetectAlign
import torch.nn as nn

class ArgMaxModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        x = self.model(x)
        race_logits = torch.argmax(x[..., 0:7], dim=1)
        gender_logits = torch.argmax(x[..., 7:9], dim=1)
        age_logits = torch.argmax(x[..., 9:18], dim=1)
        return race_logits, gender_logits, age_logits

class Inference:
    def __init__(self, model: str, checkpoint: str, det_model: str, det_checkpoint: str) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gender_labels = ['Male', 'Female']
        self.race_labels = ['White', 'Black', 'Latino Hispanic', 'East Asian', 'Southeast Asian', 'Indian', 'Middle Eastern']
        self.age_labels = ['0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70+']

        self.model = eval(model)(len(self.gender_labels) + len(self.race_labels) + len(self.age_labels))
        self.model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
        self.model = self.model.to(self.device)

        self.model.cpu()
        self.model.eval()

        self.argmax_model = ArgMaxModel(model=self.model)

        import onnx
        from onnxsim import simplify
        RESOLUTION = [
            [224,224],
        ]
        MODEL = f'attributes_fairface'
        for H, W in RESOLUTION:
            onnx_file = f"{MODEL}_1x3x{H}x{W}.onnx"
            x = torch.randn(1, 3, H, W)
            torch.onnx.export(
                self.argmax_model,
                args=(x),
                f=onnx_file,
                opset_version=13,
                input_names=['input_rgb'],
                output_names=['race_id', 'gender_id', 'age_id'],
            )
            model_onnx1 = onnx.load(onnx_file)
            model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)


            meta_data = model_onnx1.metadata_props.add()
            meta_data.key = "channel_order"
            meta_data.value = "rgb"
            meta_data = model_onnx1.metadata_props.add()
            meta_data.key = "mean"
            meta_data.value = str([0.485, 0.456, 0.406])
            meta_data = model_onnx1.metadata_props.add()
            meta_data.key = "std"
            meta_data.value = str([0.229, 0.224, 0.225])
            meta_data = model_onnx1.metadata_props.add()
            meta_data.key = "labels"
            meta_data.value = str(self.gender_labels + self.race_labels + self.age_labels)
            meta_data = model_onnx1.metadata_props.add()
            meta_data.key = "gender_labels"
            meta_data.value = str(self.gender_labels)
            meta_data = model_onnx1.metadata_props.add()
            meta_data.key = "race_labels"
            meta_data.value = str(self.race_labels)
            meta_data = model_onnx1.metadata_props.add()
            meta_data.key = "age_labels"
            meta_data.value = str(self.age_labels)
            onnx.save(model_onnx1, onnx_file)
            model_onnx2 = onnx.load(onnx_file)
            model_simp, check = simplify(model_onnx2)
            onnx.save(model_simp, onnx_file)
            model_onnx2 = onnx.load(onnx_file)
            model_simp, check = simplify(model_onnx2)
            onnx.save(model_simp, onnx_file)
            model_onnx2 = onnx.load(onnx_file)
            model_simp, check = simplify(model_onnx2)
            onnx.save(model_simp, onnx_file)

        onnx_file = f"{MODEL}_Nx3x{H}x{W}.onnx"
        x = torch.randn(1, 3, 224, 224)
        torch.onnx.export(
            self.argmax_model,
            args=(x),
            f=onnx_file,
            opset_version=13,
            input_names=['input_rgb'],
            output_names=['race_id', 'gender_id', 'age_id'],
            dynamic_axes={
                'input_rgb' : {0: 'batch'},
                'race_id' : {0: 'batch'},
                'gender_id' : {0: 'batch'},
                'age_id' : {0: 'batch'},
            }
        )
        model_onnx1 = onnx.load(onnx_file)
        model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
        meta_data = model_onnx1.metadata_props.add()
        meta_data.key = "channel_order"
        meta_data.value = "rgb"
        meta_data = model_onnx1.metadata_props.add()
        meta_data.key = "mean"
        meta_data.value = str([0.485, 0.456, 0.406])
        meta_data = model_onnx1.metadata_props.add()
        meta_data.key = "std"
        meta_data.value = str([0.229, 0.224, 0.225])
        meta_data = model_onnx1.metadata_props.add()
        meta_data.key = "labels"
        meta_data.value = str(self.gender_labels + self.race_labels + self.age_labels)
        meta_data = model_onnx1.metadata_props.add()
        meta_data.key = "gender_labels"
        meta_data.value = str(self.gender_labels)
        meta_data = model_onnx1.metadata_props.add()
        meta_data.key = "race_labels"
        meta_data.value = str(self.race_labels)
        meta_data = model_onnx1.metadata_props.add()
        meta_data.key = "age_labels"
        meta_data.value = str(self.age_labels)
        onnx.save(model_onnx1, onnx_file)
        model_onnx2 = onnx.load(onnx_file)
        model_simp, check = simplify(model_onnx2)
        onnx.save(model_simp, onnx_file)
        model_onnx2 = onnx.load(onnx_file)
        model_simp, check = simplify(model_onnx2)
        onnx.save(model_simp, onnx_file)
        model_onnx2 = onnx.load(onnx_file)
        model_simp, check = simplify(model_onnx2)
        onnx.save(model_simp, onnx_file)

        import sys
        sys.exit(0)




        self.align = FaceDetectAlign(det_model, det_checkpoint)

        self.preprocess = T.Compose([
            T.Resize((224, 224)),
            T.Lambda(lambda x: x / 255),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def visualize(self, image, dets, races, genders, ages):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        boxes = dets[:, :4].astype(int)

        for box, race, gender, age in zip(boxes, races, genders, ages):
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
            cv2.rectangle(image, (box[0], box[3] + 5), (box[2] + 20, box[3] + 50), (255, 255, 255), -1)
            cv2.putText(image, gender, (box[0], box[3] + 15), cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 0, 0), lineType=cv2.LINE_AA)
            cv2.putText(image, race, (box[0], box[3] + 30), cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 0, 0), lineType=cv2.LINE_AA)
            cv2.putText(image, age, (box[0], box[3] + 45), cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 0, 0), lineType=cv2.LINE_AA)
        return image

    def postprocess(self, preds: torch.Tensor):
        race_logits, gender_logits, age_logits = preds[:, :7].softmax(dim=1), preds[:, 7:9].softmax(dim=1), preds[:, 9:18].softmax(dim=1)
        race_preds = torch.argmax(race_logits, dim=1)
        gender_preds = torch.argmax(gender_logits, dim=1)
        age_preds = torch.argmax(age_logits, dim=1)
        return [self.race_labels[idx] for idx in race_preds], [self.gender_labels[idx] for idx in gender_preds], [self.age_labels[idx] for idx in age_preds]

    def __call__(self, img_path: Union[str, np.ndarray]):
        faces, dets, image = self.align.detect_and_align_faces(img_path, (112, 112))
        if faces is None:
            return cv2.cvtColor(image[0], cv2.COLOR_RGB2BGR)

        pfaces = self.preprocess(faces.permute(0, 3, 1, 2)).to(self.device)

        with torch.inference_mode():
            preds = self.model(pfaces).detach().cpu()
        races, genders, ages = self.postprocess(preds)

        image = self.visualize(image[0], dets[0], races, genders, ages)
        return image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='assets/asian_american.jpg')
    parser.add_argument('--model', type=str, default='FairFace')
    parser.add_argument('--checkpoint', type=str, default='/home/sithu/checkpoints/facialattributes/fairface/res34_fairface.pth')
    parser.add_argument('--det_model', type=str, default='RetinaFace')
    parser.add_argument('--det_checkpoint', type=str, default='/home/sithu/checkpoints/FR/retinaface/mobilenet0.25_Final.pth')
    args = vars(parser.parse_args())
    source = args.pop('source')
    file_path = Path(source)

    inference = Inference(**args)

    if file_path.is_file():
        if file_path.suffix in ['.mp4', '.avi', '.m4v']:
            reader = VideoReader(str(file_path))
            writer = VideoWriter(f"{str(file_path).split('.', maxsplit=1)[0]}_out.mp4", reader.fps)

            for frame in tqdm(reader):
                image = inference(frame)
                writer.update(image[:, :, ::-1])
            writer.write()
        else:
            image = inference(str(file_path))
            image = Image.fromarray(image[:, :, ::-1]).convert('RGB')
            image.show()

    elif str(file_path) == 'webcam':
        stream = WebcamStream(0)
        fps = FPS()

        for frame in stream:
            fps.start()
            frame = inference(frame)
            fps.stop()
            cv2.imshow('frame', frame)

    else:
        raise FileNotFoundError(f"The following file does not exist: {str(file_path)}")
