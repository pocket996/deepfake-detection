"""Predict on one photo or video without modifying the source media."""
import argparse
import json
import torch
import cv2
from core import HERE, Predictor, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input')
    parser.add_argument('--checkpoint', help='Default: fine-tuned model when available, otherwise frozen baseline')
    parser.add_argument('--face-crop', action='store_true', help='Input image already is a square face crop')
    parser.add_argument('--output')
    parser.add_argument('--device', choices=['cpu', 'cuda'])
    parser.add_argument('--task', choices=['auto', 'generation', 'manipulation'], default='auto',
                        help='auto: photos -> generation, videos -> manipulation')
    args = parser.parse_args()
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    if args.checkpoint:
        if args.task == 'generation':
            from face_analysis import GenerationPredictor
            predictor = GenerationPredictor(args.checkpoint, args.device)
        else:
            predictor = Predictor(args.checkpoint, args.device)
        result = predictor.predict(args.input, args.face_crop)
    else:
        from face_analysis import FaceAnalyzer
        result = FaceAnalyzer(args.device).predict(args.input, face_crop=args.face_crop, task=args.task)
    if args.output:
        save_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
