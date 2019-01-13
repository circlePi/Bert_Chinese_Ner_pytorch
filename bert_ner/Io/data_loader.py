import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from preprocessing.data_processor import MyPro, convert_examples_to_features
import config.args as args
from util.Logginger import init_logger

logger = init_logger("bert_ner",logging_path=args.log_path)


def init_params():
    processors = {"bert_ner": MyPro}
    task_name = args.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))
    processor = processors[task_name]()
    tokenizer = BertTokenizer(vocab_file=args.VOCAB_FILE)
    return processor, tokenizer


def create_batch_iter(mode):
    """构造迭代器"""
    processor, tokenizer = init_params()
    if mode == "train":
        examples = processor.get_train_examples(args.data_dir)

        num_train_steps = int(
            len(examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

        batch_size = args.train_batch_size

        logger.info("  Num steps = %d", num_train_steps)

    elif mode == "dev":
        examples = processor.get_dev_examples(args.data_dir)
        batch_size = args.eval_batch_size
    else:
        raise ValueError("Invalid mode %s" % mode)

    label_list = processor.get_labels()

    # 特征
    features = convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer)

    logger.info("  Num examples = %d", len(examples))
    logger.info("  Batch size = %d", batch_size)


    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    all_output_mask = torch.tensor([f.output_mask for f in features], dtype=torch.long)

    # 数据集
    data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_output_mask)

    if mode == "train":
        sampler = RandomSampler(data)
    elif mode == "dev":
        sampler = SequentialSampler(data)
    else:
        raise ValueError("Invalid mode %s" % mode)

    # 迭代器
    iterator = DataLoader(data, sampler=sampler, batch_size=batch_size)

    if mode == "train":
        return iterator, num_train_steps
    elif mode == "dev":
        return iterator
    else:
        raise ValueError("Invalid mode %s" % mode)


