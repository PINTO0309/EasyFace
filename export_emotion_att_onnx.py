import torch
import argparse
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from torchvision import transforms as T
from typing import Union
import ast
import torch.nn as nn

from easyface.emotion.models import *
from easyface.attributes.models import *
from easyface.utils.visualize import draw_box_and_landmark, show_image
from easyface.utils.io import WebcamStream, VideoReader, VideoWriter, FPS
from detect_align import FaceDetectAlign

class ArgMaxModel(nn.Module):
    def __init__(self, model_att, model_emotion):
        super().__init__()
        self.model_att = model_att
        self.model_emotion = model_emotion

    def forward(self, x):
        # attributes
        y = self.model_att(x)
        race_logits = torch.argmax(y[..., 0:7], dim=1, keepdim=True)
        gender_logits = torch.argmax(y[..., 7:9], dim=1, keepdim=True)
        age_logits = torch.argmax(y[..., 9:18], dim=1, keepdim=True)
        # emotion
        z = self.model_emotion(x)
        emotion_logits = torch.argmax(z, dim=1, keepdim=True)

        # merge
        all_logits = torch.cat([race_logits, gender_logits, age_logits, emotion_logits], dim=1)

        return all_logits

class Inference:
    def __init__(self, model_emotion: str, dataset: str, checkpoint_emotion: str, model_att: str, checkpoint_att: str, det_model: str, det_checkpoint: str) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.labels = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger']
        assert dataset in ['AffectNet7', 'AffectNet8', 'RAFDB']
        if dataset == 'AffectNet8':
            self.labels.append('Contempt')
        elif dataset == 'RAFDB':
            self.labels = ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral']
        self.model_emotion = eval(model_emotion)(len(self.labels))
        self.model_emotion.load_checkpoint(checkpoint_emotion)
        self.model_emotion.cpu()
        self.model_emotion.eval()


        self.gender_labels = ['Male', 'Female']
        self.race_labels = ['White', 'Black', 'Latino Hispanic', 'East Asian', 'Southeast Asian', 'Indian', 'Middle Eastern']
        self.age_labels = ['0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70+']
        self.model_att = eval(model_att)(len(self.gender_labels) + len(self.race_labels) + len(self.age_labels))
        self.model_att.load_state_dict(torch.load(checkpoint_att, map_location='cpu'))
        self.model_att = self.model_att.to(self.device)
        self.model_att.cpu()
        self.model_att.eval()

        self.argmax_model = ArgMaxModel(model_att=self.model_att, model_emotion=self.model_emotion)

        import onnx
        from onnxsim import simplify
        RESOLUTION = [
            [224,224],
        ]
        MODEL = f'fairdan_{dataset.lower()}'
        for H, W in RESOLUTION:
            onnx_file = f"{MODEL}_1x3x{H}x{W}.onnx"
            x = torch.randn(1, 3, H, W)
            torch.onnx.export(
                self.argmax_model,
                args=(x),
                f=onnx_file,
                opset_version=13,
                input_names=['input_rgb'],
                output_names=['raceid_genderid_ageid_emotionid'],
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
            meta_data.key = "gender_labels"
            meta_data.value = str(self.gender_labels)
            meta_data = model_onnx1.metadata_props.add()
            meta_data.key = "race_labels"
            meta_data.value = str(self.race_labels)
            meta_data = model_onnx1.metadata_props.add()
            meta_data.key = "age_labels"
            meta_data.value = str(self.age_labels)
            meta_data = model_onnx1.metadata_props.add()
            meta_data.key = "emotion_labels"
            meta_data.value = str(self.labels)

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
            output_names=['raceid_genderid_ageid_emotionid'],
            dynamic_axes={
                'input_rgb' : {0: 'batch'},
                'raceid_genderid_ageid_emotionid': {0: 'batch'},
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

        meta_data = model_onnx1.metadata_props.add()
        meta_data.key = "gender_labels"
        meta_data.value = str(self.gender_labels)
        meta_data = model_onnx1.metadata_props.add()
        meta_data.key = "race_labels"
        meta_data.value = str(self.race_labels)
        meta_data = model_onnx1.metadata_props.add()
        meta_data.key = "age_labels"
        meta_data.value = str(self.age_labels)
        meta_data = model_onnx1.metadata_props.add()
        meta_data.key = "emotion_labels"
        meta_data.value = str(self.labels)

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
            T.Lambda(lambda x: x / 255),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def visualize(self, image, dets, labels, scores):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        boxes, landmarks = dets[:, :4].astype(int), dets[:, 5:].astype(int)

        for box, score, label, landmark in zip(boxes, scores, labels, landmarks):
            text = f"{label}: {int(score*100):2d}%"
            draw_box_and_landmark(image, box, text, landmark, draw_lmks=False)
        return image

    def postprocess(self, preds: torch.Tensor):
        preds = preds.softmax(dim=1)
        probs, idxs = torch.max(preds, dim=1)
        return [self.labels[idx] for idx in idxs], probs.tolist()

    def __call__(self, img_path: Union[str, np.ndarray]):
        faces, dets, image = self.align.detect_and_crop_faces(img_path, (224, 224))
        if faces is None:
            return cv2.cvtColor(image[0], cv2.COLOR_RGB2BGR)

        pfaces = self.preprocess(faces.permute(0, 3, 1, 2)).to(self.device)

        with torch.inference_mode():
            preds = self.model(pfaces)[0].detach().cpu()
        labels, scores = self.postprocess(preds)

        image = self.visualize(image[0], dets[0], labels, scores)
        return image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='assets/test.jpg')

    parser.add_argument('--model_emotion', type=str, default='DAN')
    parser.add_argument('--dataset', type=str, default='AffectNet8')
    parser.add_argument('--checkpoint_emotion', type=str, default='/home/sithu/checkpoints/face_emotion/affecnet8_epoch5_acc0.6209.pth')

    parser.add_argument('--model_att', type=str, default='FairFace')
    parser.add_argument('--checkpoint_att', type=str, default='/home/sithu/checkpoints/facialattributes/fairface/res34_fairface.pth')

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
