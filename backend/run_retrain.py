"""Lightweight retrain + evaluate wrapper.

Usage (PowerShell):
  python run_retrain.py --data "..\Dataset\Garbage classification" --epochs 10 --batch 16

This script imports train.train and evaluate.evaluate to run one after another.
"""
import argparse
import os
import sys

# ensure backend folder on sys.path
root = os.path.abspath(os.path.dirname(__file__))
if root not in sys.path:
    sys.path.insert(0, root)

from train import train as do_train
from evaluate import evaluate as do_evaluate

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Path to dataset root ("dataset")')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--model', default='efficientnet_b0', choices=['efficientnet_b0', 'efficientnet_b3'])
    parser.add_argument('--out', default='backend/model.pth')
    parser.add_argument('--eval_batch', type=int, default=16)
    args = parser.parse_args()

    print('Starting training...')
    do_train(args.data, model_name=args.model, epochs=args.epochs, batch_size=args.batch, out_path=args.out)
    print('Training finished — running evaluation')
    do_evaluate(args.data, args.out, batch_size=args.eval_batch)
    print('Retrain+evaluate complete')
