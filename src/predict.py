"""Usage: predict.py [-m MODEL] [-s BS] [-d DECODE] [-b BEAM] [IMAGE ...]

-h, --help    show this
-m MODEL     model file [default: ./checkpoints/crnn_synth90k.pt]
-s BS       batch size [default: 256]
-d DECODE    decode method (greedy, beam_search or prefix_beam_search) [default: beam_search]
-b BEAM   beam size [default: 10]

"""
from docopt import docopt
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from config import common_config as config
from dataset import Synth90kDataset, synth90k_collate_fn, KalapaDataset
from model import CRNN
from ctc_decoder import ctc_decode
import os
import csv

def predict(crnn, dataloader, label2char, decode_method, beam_size):
    crnn.eval()
    pbar = tqdm(total=len(dataloader), desc="Predict")

    all_preds = []
    with torch.no_grad():
        for data in dataloader:
            device = 'cuda' if next(crnn.parameters()).is_cuda else 'cpu'

            images = data.to(device)

            logits = crnn(images)
            log_probs = torch.nn.functional.log_softmax(logits, dim=2)

            preds = ctc_decode(log_probs, method=decode_method, beam_size=beam_size,
                               label2char=label2char)
            all_preds += preds

            pbar.update(1)
        pbar.close()

    return all_preds


def show_result(paths, preds):
    print('\n===== result =====')
    for path, pred in zip(paths, preds):
        text = ''.join(pred)
        print(f'{path} > {text}')


def main():
    arguments = docopt(__doc__)

    # images = arguments['IMAGE']
    reload_checkpoint = arguments['-m']
    batch_size = int(arguments['-s'])
    decode_method = arguments['-d']
    beam_size = int(arguments['-b'])

    img_height = config['img_height']
    img_width = config['img_width']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')
    
    root_folder = 'OCR/public_test/images'
    folders = sorted(os.listdir(root_folder), key=lambda x: int(x))
    
    images = []
    for folder in folders:
        folder = os.path.join(root_folder, folder)
        sorted_files = sorted(os.listdir(folder), key=lambda x: int(x.split('.')[0]))
        images += [os.path.join(folder, file) for file in sorted_files]

    # predict_dataset = Synth90kDataset(paths=images,
                                    #   img_height=img_height, img_width=img_width)
    predict_dataset = KalapaDataset(paths=images,
                                      img_height=img_height, img_width=img_width)
    predict_loader = DataLoader(
        dataset=predict_dataset,
        batch_size=batch_size,
        shuffle=False)

    # num_class = len(Synth90kDataset.LABEL2CHAR) + 1
    num_class = len(KalapaDataset.LABEL2CHAR) + 1
    
    crnn = CRNN(1, img_height, img_width, num_class,
                map_to_seq_hidden=config['map_to_seq_hidden'],
                rnn_hidden=config['rnn_hidden'],
                leaky_relu=config['leaky_relu'])
    crnn.load_state_dict(torch.load(reload_checkpoint, map_location=device))
    crnn.to(device)

    # preds = predict(crnn, predict_loader, Synth90kDataset.LABEL2CHAR,
    #                 decode_method=decode_method,
    #                 beam_size=beam_size)

    preds = predict(crnn, predict_loader, KalapaDataset.LABEL2CHAR,
                    decode_method=decode_method,
                    beam_size=beam_size)

    contents = [image.split("OCR/public_test/images/")[-1] for image in images]
    show_result(contents, preds)

    preds = [''.join(pred) for pred in preds]
   

    data = [['id', 'answer']]
    data.extend(zip(contents, preds))

    filename = 'sample_submission.csv'

    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

    print(f"File '{filename}' đã được tạo thành công.")


if __name__ == '__main__':
    main()
