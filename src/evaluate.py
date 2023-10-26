import torch
from torch.utils.data import DataLoader
from torch.nn import CTCLoss
from tqdm import tqdm

from dataset import Synth90kDataset, synth90k_collate_fn, KalapaDataset
from model import CRNN
from ctc_decoder import ctc_decode
from config import evaluate_config as config

torch.backends.cudnn.enabled = False

def edit_distance(preds, reals):
    m = len(preds)
    n = len(reals)

    # Tạo ma trận DP với kích thước (m+1) x (n+1)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Khởi tạo giá trị ban đầu
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # Tính toán edit distance
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if preds[i - 1] == reals[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1

    return dp[m][n]

def evaluate(crnn, dataloader, criterion,
             max_iter=None, decode_method='beam_search', beam_size=10):
    crnn.eval()

    tot_count = 0
    tot_loss = 0
    tot_correct = 0
    tot_distance = 0
    wrong_cases = []

    pbar_total = max_iter if max_iter else len(dataloader)
    pbar = tqdm(total=pbar_total, desc="Evaluate")

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            if max_iter and i >= max_iter:
                break
            device = 'cuda' if next(crnn.parameters()).is_cuda else 'cpu'

            images, targets, target_lengths = [d.to(device) for d in data]

            logits = crnn(images)
            log_probs = torch.nn.functional.log_softmax(logits, dim=2)

            batch_size = images.size(0)
            input_lengths = torch.LongTensor([logits.size(0)] * batch_size)

            loss = criterion(log_probs, targets, input_lengths, target_lengths)

            preds = ctc_decode(log_probs, method=decode_method, beam_size=beam_size)
            reals = targets.cpu().numpy().tolist()
            target_lengths = target_lengths.cpu().numpy().tolist()

            tot_count += batch_size
            tot_loss += loss.item()
            tot_distance += edit_distance(preds, reals)

            target_length_counter = 0
            for pred, target_length in zip(preds, target_lengths):
                real = reals[target_length_counter:target_length_counter + target_length]
                target_length_counter += target_length
                if pred == real:
                    tot_correct += 1
                else:
                    wrong_cases.append((real, pred))

            pbar.update(1)
        pbar.close()

    evaluation = {
        'loss': tot_loss / tot_count,
        'acc': tot_correct / tot_count,
        'distance': tot_distance / tot_count,
        'wrong_cases': wrong_cases
    }
    return evaluation


def main():
    eval_batch_size = config['eval_batch_size']
    cpu_workers = config['cpu_workers']
    reload_checkpoint = config['reload_checkpoint']

    img_height = config['img_height']
    img_width = config['img_width']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    # test_dataset = Synth90kDataset(root_dir=config['data_dir'], mode='test',
    #                                img_height=img_height, img_width=img_width)
    test_dataset = KalapaDataset(root_dir=config['data_dir'], mode='test',
                                   img_height=img_height, img_width=img_width)

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=cpu_workers,
        collate_fn=synth90k_collate_fn)

    # num_class = len(Synth90kDataset.LABEL2CHAR) + 1
    num_class = len(KalapaDataset.LABEL2CHAR) + 1
    crnn = CRNN(1, img_height, img_width, num_class,
                map_to_seq_hidden=config['map_to_seq_hidden'],
                rnn_hidden=config['rnn_hidden'],
                leaky_relu=config['leaky_relu'])
    reload_checkpoint = 'checkpoints/crnn_044000_convnextv2_loss0.0.pt'
    crnn.load_state_dict(torch.load(reload_checkpoint, map_location=device))
    crnn.to(device)

    criterion = CTCLoss(reduction='sum')
    criterion.to(device)

    evaluation = evaluate(crnn, test_loader, criterion,
                          decode_method=config['decode_method'],
                          beam_size=config['beam_size'])
    print('test_evaluation: loss={loss}, acc={acc}, edit_distance={distance}'.format(**evaluation))


if __name__ == '__main__':
    main()
