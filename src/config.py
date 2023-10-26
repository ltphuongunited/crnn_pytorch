
common_config = {
    'data_dir': 'OCR/training_data',
    # 'img_width': 100,
    # 'img_height': 32,
    'img_width': 1024,
    'img_height': 64,
    # 'map_to_seq_hidden': 64,
    'map_to_seq_hidden': 256,
    'rnn_hidden': 256,
    'leaky_relu': False,
}

train_config = {
    'epochs': 10000,
    'train_batch_size': 128,
    'eval_batch_size': 512,
    'lr': 0.0001,
    'show_interval': 10,
    'valid_interval': 500,
    'save_interval': 2000,
    # 'valid_interval': 5,
    # 'save_interval': 5,
    'cpu_workers': 16,
    'reload_checkpoint': None,
    'valid_max_iter': 100,
    'decode_method': 'greedy',
    'beam_size': 10,
    'checkpoints_dir': 'checkpoints/'
}
train_config.update(common_config)

evaluate_config = {
    'eval_batch_size': 512,
    'cpu_workers': 16,
    'reload_checkpoint': 'checkpoints/crnn.pt',
    'decode_method': 'beam_search',
    'beam_size': 10,
}
evaluate_config.update(common_config)
