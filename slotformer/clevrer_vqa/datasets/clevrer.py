import os
import copy
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

from nerv.utils import load_obj, VideoReader, strip_suffix, read_img

from .utils import np_stack, np_concat, torch_stack, CLEVRTransforms


class CLEVRERVQADataset(Dataset):
    """Dataset for loading CLEVRER VQA videos and QA pairs.

    Args:
        vocab_file (str): path to pre-computed dataset vocabulary.
        max_question_len (int): pad the questions to this length.
        max_choice_len (int): pad the choices to this length.
    """

    def __init__(
        self,
        data_root,
        vocab_file,
        clevrer_transforms,
        split='train',
        max_n_objects=6,
        video_len=128,
        n_sample_frames=25,
        max_question_len=20,
        max_choice_len=12,
    ):
        super().__init__()
        self.data_root = data_root
        self.split = split
        assert self.split in ['train', 'val', 'test']
        self.video_path = os.path.join(data_root, 'videos', split)
        # self.anno_path = os.path.join(data_root, 'annotations', split)

        self.clevrer_transforms = clevrer_transforms
        self.max_n_objects = max_n_objects
        self.video_len = video_len
        self.n_sample_frames = n_sample_frames
        self.frame_offset = video_len // n_sample_frames

        # all video paths
        self.files = self.get_files()  # {`video_fn`: `video_path`}
        self.num_videos = len(self.files)
        self.valid_idx = self.get_sample_idx()  # {`video_fn`: `valid_idx`}

        # question, answer, choice pairs
        self.vocab_file = vocab_file
        self.max_question_len = max_question_len
        self.max_choice_len = max_choice_len
        self.q_subtype2id = {
            'descriptive': 0,
            'explanatory': 1,
            'predictive': 2,
            'counterfactual': 3,
        }
        self.cls_questions, self.mc_questions = self.get_questions()
        self.num_cls_questions = len(self.cls_questions)
        self.num_mc_questions = len(self.mc_questions)
        """
        Shared items:
            - scene_index: the index of the scene, e.g. 15000
            - video_filename: video_filename, e.g. 'video_00000.mp4'
            - question_id: the id of this question (in this scene)
            - q_subtype: the id of the question's subtype (4 types in total)
            - q_tokens: 1d int array of question str tokens
            - q_pad_mask: 1d boolean array of question pad mask, True is pad
            - raw_question: question str

        cls_questions specific:
            - raw_answer: answer str
            - a_label: converted answer label

        mc_questions:
            - choice_id: a list of choice id number
            - raw_choices: a list of choice str
            - raw_answers: a list of answer str
            - c_tokens: a list of choice tokens, 1d int array
            - c_pad_mask: a list of pad masks, 1d boolean array
            - c_label: a list of True/False for each choice
        """

        # whether to include image in loaded data dict
        self.load_frames = True

    def _get_question_dict(self, idx):
        """Returns question dict and question type."""
        if idx < self.num_cls_questions:
            return copy.deepcopy(self.cls_questions[idx]), 0
        return copy.deepcopy(self.mc_questions[idx -
                                               self.num_cls_questions]), 1

    def _get_frames(self, video_fn):
        """Get and pre-process video frames."""
        video_path = self.files[video_fn]
        start_idx = np.random.choice(self.valid_idx[video_fn])
        frame_dir = strip_suffix(video_path)
        # videos are not converted to frames, read from mp4 file
        if not os.path.isdir(frame_dir):
            cap = VideoReader(video_path)
            frames = [
                cap.get_frame(start_idx + n * self.frame_offset)
                for n in range(self.n_sample_frames)
            ]
        else:
            # empty video
            if len(os.listdir(frame_dir)) != self.video_len:
                raise ValueError
            # read from jpg images
            filename = os.path.join(frame_dir, '{:06d}.jpg')
            frames = [
                read_img(filename.format(start_idx + n * self.frame_offset))
                for n in range(self.n_sample_frames)
            ]
        if any(frame is None for frame in frames):
            raise ValueError
        frames = [
            self.clevrer_transforms(Image.fromarray(img).convert('RGB'))
            for img in frames
        ]
        return torch.stack(frames, dim=0), start_idx  # [T, C, H, W]

    def _rand_another(self, idx):
        """Random get another sample when encountering loading error."""
        if self._get_question_dict(idx)[1] == 0:
            another_idx = np.random.randint(0, self.num_cls_questions)
        else:
            another_idx = np.random.randint(self.num_cls_questions, len(self))
        return self.__getitem__(another_idx)

    def __getitem__(self, idx):
        """Data dict:
            - scene_index: int
            - question_id: int
            - video: [T, C, H, W] CLEVRER video frames
            - start_idx: the frame index to start uniform sample frames
            - q_type: 0 as cls q while 1 as mc q
            - q_subtype: 0, 1, 2, 3 as 4 subtypes of questions
            - q_tokens:
                in cls question, int array tokens of question;
                in mc question, list of int array tokens of question+choice
            - q_pad_mask:
                in cls question, boolean array tokens of question;
                in mc question, list of boolean array tokens of question+choice
            - a_label:
                in cls question, an int label;
                in mc question, a {0, 1} label array
            - mc_flag (only in mc question):
                an all_zeros array of shape (num_choices, )
            - choice_id (only in mc question):
                a list of int number of shape (num_choices, )
        """
        question, q_type = self._get_question_dict(idx)
        video_fn = question['video_filename']
        q_dict = {
            'scene_index': question['scene_index'],
            'question_id': question['question_id'],
            'q_subtype': question['q_subtype'],
            'q_tokens': question['q_tokens'],
            'q_pad_mask': question['q_pad_mask'],
            'q_type': q_type,
        }
        # cls question
        if q_type == 0:
            q_dict['a_label'] = question['a_label']
        # mc question
        else:
            # we need to concat question and choice as model input
            q_dict['q_tokens'] = np.stack([
                np.concatenate([q_dict['q_tokens'], c_tokens])
                for c_tokens in question['c_tokens']
            ])
            q_dict['q_pad_mask'] = np.stack([
                np.concatenate([q_dict['q_pad_mask'], c_pad_mask])
                for c_pad_mask in question['c_pad_mask']
            ])
            q_dict['a_label'] = np.stack(question['c_label']).astype(np.int32)
            q_dict['mc_flag'] = np.zeros_like(q_dict['a_label'])
            q_dict['mc_choice_id'] = np.stack(question['choice_id'])

        if self.load_frames:
            try:
                q_dict['video'], start_idx = self._get_frames(video_fn)
            # empty video
            except ValueError:
                return self._rand_another(idx)
        else:
            start_idx = np.random.choice(self.valid_idx[video_fn])
        # `start_idx` is just for child class usage
        # i.e. when loading frame_embs, we need to sample the same frames
        q_dict['start_idx'] = start_idx
        return q_dict

    def __len__(self):
        return self.num_cls_questions + self.num_mc_questions

    def _tokenize_text(self, q_str, pad_num):
        """Convert a question str to a 1d np array and do the padding."""
        q = q_str.lower().replace('?', '').split(' ')
        q_tokens = [self.q_vocab[word] for word in q if word]  # eliminate ''
        pad_mask = np.ones(pad_num).astype(np.bool)
        pad_mask[:len(q_tokens)] = False
        q_tokens += [
            self.q_vocab['PAD'] for _ in range(pad_num - len(q_tokens))
        ]
        return np.array(q_tokens).astype(np.int32), pad_mask

    def get_files(self):
        """Get dict with video_filename as key and full video path as value."""
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        cache = os.path.join(cur_dir, 'cache/CLEVRER_video_fn2video_path.json')
        video_paths = load_obj(cache)[self.split]
        print(f'Loading {len(video_paths)} video files from json file...')
        return video_paths

    def get_sample_idx(self):
        """Get dict with video_filename as key and valid start_idx as value."""
        video_filenames = sorted(list(self.files.keys()))
        valid_idx = {}
        for video_filename in video_filenames:
            # simply use random uniform sampling
            max_start_idx = self.video_len - \
                (self.n_sample_frames - 1) * self.frame_offset
            valid_idx[video_filename] = list(range(max_start_idx))
        return valid_idx

    def get_questions(self):
        """Get question and answer pairs.

        Descriptive and multiple choice questions are stored separately.
        """
        # load vocabs which are two word2idx dicts for question and answer
        vocabs = load_obj(self.vocab_file)
        self.q_vocab = vocabs['q_vocab']
        self.answer2label = vocabs['a_vocab']
        self.label2answer = {v: k for k, v in self.answer2label.items()}

        question_file = os.path.join(self.data_root, 'questions',
                                     f'{self.split}.json')
        json_question = load_obj(question_file)
        cls_questions, mc_questions = [], []
        for questions in json_question:  # for each scene
            for q in questions['questions']:  # for each question in this scene
                q_dict = {
                    'scene_index': questions['scene_index'],  # int
                    'video_filename': questions['video_filename'],  # str
                    'question_id': q['question_id'],  # int
                    'raw_question': q['question'],  # str
                    'q_subtype': self.q_subtype2id[q['question_type']],  # int
                }
                # cls question
                if q['question_type'] == 'descriptive':
                    # directly pad choice as zeros
                    # so that q_tokens in cls and mc will have same length
                    q_tokens, q_pad_mask = self._tokenize_text(
                        q['question'],
                        self.max_question_len + self.max_choice_len)
                    q_dict['q_tokens'] = q_tokens
                    q_dict['q_pad_mask'] = q_pad_mask
                    if 'answer' in q:
                        q_dict['raw_answer'] = q['answer']
                        q_dict['a_label'] = int(self.answer2label[q['answer']])
                    else:
                        # test set
                        q_dict['a_label'] = -1
                    cls_questions.append(q_dict)
                # multiple choice question
                else:
                    q_tokens, q_pad_mask = self._tokenize_text(
                        q['question'], self.max_question_len)
                    q_dict['q_tokens'] = q_tokens
                    q_dict['q_pad_mask'] = q_pad_mask
                    q_dict['raw_choices'], q_dict['raw_answers'] = [], []
                    q_dict['c_tokens'], q_dict['c_pad_mask'] = [], []
                    q_dict['choice_id'], q_dict['c_label'] = [], []
                    for choice in q['choices']:
                        q_dict['choice_id'].append(choice['choice_id'])
                        q_dict['raw_choices'].append(choice['choice'])
                        if 'answer' in choice:
                            q_dict['raw_answers'].append(choice['answer'])
                            q_dict['c_label'].append(
                                choice['answer'] == 'correct')
                        else:
                            q_dict['raw_answers'].append('')
                            q_dict['c_label'].append(True)
                        c_tokens, c_pad_mask = self._tokenize_text(
                            choice['choice'], self.max_choice_len)
                        q_dict['c_tokens'].append(c_tokens)
                        q_dict['c_pad_mask'].append(c_pad_mask)
                    mc_questions.append(q_dict)

        return cls_questions, mc_questions

    def get_answer_from_label(self, answer_labels):
        """Get answer text from answer labels."""
        assert isinstance(answer_labels, np.ndarray)
        ori_shape = answer_labels.shape
        answer_labels = answer_labels.flatten()
        answer_texts = [self.label2answer[lbl] for lbl in answer_labels]
        return np.array(answer_texts).reshape(ori_shape)

    def get_qa_text(self, idx):
        """Get raw text of a question answer pair."""
        question, q_type = self._get_question_dict(idx)
        q_str = question['raw_question']
        if q_type == 0:
            return q_str, question['raw_answer']
        else:
            return q_str, question['raw_choices'], question['raw_answers']


class CLEVRERSlotsVQADataset(CLEVRERVQADataset):
    """Dataset for loading CLEVRER VQA video embs and QA pairs.

    Args:
        video_slots (dict): pre-computed {video_fn: slots}.
        shuffle_obj (bool): whether to shuffle the order of object embs.
    """

    def __init__(
        self,
        data_root,
        video_slots,
        vocab_file,
        clevrer_transforms,
        split='train',
        max_n_objects=6,
        video_len=128,
        n_sample_frames=25,
        max_question_len=20,
        max_choice_len=12,
        shuffle_obj=False,
    ):
        super().__init__(
            data_root=data_root,
            vocab_file=vocab_file,
            clevrer_transforms=clevrer_transforms,
            split=split,
            max_n_objects=max_n_objects,
            video_len=video_len,
            n_sample_frames=n_sample_frames,
            max_question_len=max_question_len,
            max_choice_len=max_choice_len,
        )

        # a dict with video_filename as key and slots as value
        self.video_slots = video_slots

        self.load_frames = False
        self.shuffle_obj = shuffle_obj

    def _get_slots(self, idx, start_idx):
        """Extract and potentially shuffle object embs."""
        question, _ = self._get_question_dict(idx)
        video_fn = question['video_filename']
        assert video_fn in self.video_slots.keys()
        embs = self.video_slots[video_fn]  # [T, N, C]
        # for predictive questions, we use later frames
        # `embs.shape[0] > 150` means the load slots are explicitly unrolled
        # though we're using GT question types, this is not a problem
        # because in DCL/VRDP, their question parser can achieve ~100% accuracy
        # which is the same as ours
        if question['q_subtype'] == 2 and embs.shape[0] > 150:
            start_idx += (embs.shape[0] - self.video_len)
        sample_idx = np.array([
            start_idx + n * self.frame_offset
            for n in range(self.n_sample_frames)
        ])
        embs = embs[sample_idx]
        if self.shuffle_obj:
            idx = torch.randperm(embs.shape[1]).long().numpy()
            embs = embs[:, idx]
        return torch.from_numpy(embs)

    def __getitem__(self, idx):
        """Data dict (added compared to its super class):
            - video_emb: [T, N, C] pre-computed img embeddings from frames
        """
        data_dict = super().__getitem__(idx)
        try:
            data_dict['video_emb'] = self._get_slots(idx,
                                                     data_dict['start_idx'])
        # empty video --> no frame embs
        except AssertionError:
            if self.split != 'test':
                return self._rand_another(idx)
            # at test time, we need to pad the question_id for submission
            # so we keep all question_id unchanged, just pad the video_emb
            video_emb = self._rand_another(idx)['video_emb']
            data_dict['video_emb'] = video_emb
        return data_dict


def clevrer_collate_fn(list_data):
    """Special collate_fn for CLEVRER QA datasets.

    We need to put cls questions and mc questions to two subset and batching.

    Args:
        list_data (List[dict]): each dict with keys:
            - scene_index: int
            - question_id: int
            - video (optional): [T, C, H, W] CLEVRER video frames
            - video_emb (optional): [T, N, C] pre-computed frame embeddings
            - start_idx: the frame index to start uniform sample frames
            - q_type: 0 as cls q while 1 as mc q
            - q_subtype: 0, 1, 2, 3 as 4 subtypes of questions
            - q_tokens:
                in cls question, int array tokens of question [L];
                in mc question, list of tokens of question+choice [n, L]
            - q_pad_mask:
                in cls question, boolean array tokens of question [L];
                in mc question, list of tokens of question+choice [n, L]
            - a_label:
                in cls question, an int label;
                in mc question, a {0, 1} label array [n]
            - mc_flag (only in mc question):
                an all_zeros array of shape (num_choices, )
            - mc_choice_id (only in mc question):
                a list of int number of shape (num_choices, )

    Returns:
        dict: with keys:
            - scene_index: [B]
            - question_id: [B]

            - cls_/mc_video (optional): [B1, T, C, H, W]
            - cls_/mc_video_emb (optional): [B1, T, N, C]

            - cls_q_tokens: [B1, L]
            - cls_q_pad_mask: [B1, L]
            - cls_label: [B1]

            - mc_subtype: [B2]
            - mc_q_tokens: [B2 n, L], concated along num_choices dim
            - mc_q_pad_mask: [B2 n, L]
            - mc_label: [B2 n]
            - mc_flag: [B2 n], e.g. [0, 0, 0, 1, 1, 1, 1, 2, 2, ...]
                indicating which first_dim_idx corresponds to which video
            - mc_choice_id: [B2 n], e.g. [0, 1, 2, 0, 1, 2, 3, 0, 1, ...]
    """
    cls_data = [data for data in list_data if data['q_type'] == 0]
    mc_data = [data for data in list_data if data['q_type'] == 1]
    # pack cls and mc questions separately
    num_mc = len(mc_data)
    mc_flag = np_concat([mc_data[i]['mc_flag'] + i for i in range(num_mc)])
    batch_data = {
        'scene_index': np_stack([data['scene_index'] for data in list_data]),
        'question_id': np_stack([data['question_id'] for data in list_data]),
        'cls_q_tokens': np_stack([data['q_tokens'] for data in cls_data]),
        'cls_q_pad_mask': np_stack([data['q_pad_mask'] for data in cls_data]),
        'cls_label': np_stack([data['a_label'] for data in cls_data]),
        'mc_subtype': np_stack([data['q_subtype'] for data in mc_data]),
        'mc_q_tokens': np_concat([data['q_tokens'] for data in mc_data]),
        'mc_q_pad_mask': np_concat([data['q_pad_mask'] for data in mc_data]),
        'mc_label': np_concat([data['a_label'] for data in mc_data]),
        'mc_flag': mc_flag,
        'mc_choice_id': np_concat([data['mc_choice_id'] for data in mc_data]),
    }
    batch_data = {k: torch.from_numpy(v) for k, v in batch_data.items()}
    if 'video' in list_data[0].keys():
        batch_data['cls_video'] = torch_stack(
            [data['video'] for data in cls_data])
        batch_data['mc_video'] = torch_stack(
            [data['video'] for data in mc_data])
    if 'video_emb' in list_data[0].keys():
        batch_data['cls_video_emb'] = torch_stack(
            [data['video_emb'] for data in cls_data])
        batch_data['mc_video_emb'] = torch_stack(
            [data['video_emb'] for data in mc_data])
    return batch_data


def build_clevrer_slots_vqa_dataset(params, test_set=False):
    """Build CLEVRER VQA dataset with pre-computed slots."""
    video_slots = load_obj(params.slots_root)
    args = dict(
        data_root=params.data_root,
        video_slots=None,
        vocab_file=params.vocab_file,
        clevrer_transforms=CLEVRTransforms((128, 128)),
        split=None,
        max_n_objects=params.max_n_objects,
        video_len=128,
        n_sample_frames=params.n_sample_frames,
        max_question_len=params.max_question_len,
        max_choice_len=params.max_choice_len,
        shuffle_obj=params.shuffle_obj,
    )

    if test_set:
        args['split'] = 'test'
        args['video_slots'] = video_slots['test']
        args['shuffle_obj'] = False
        test_dataset = CLEVRERSlotsVQADataset(**args)
        return test_dataset

    args['split'] = 'val'
    args['video_slots'] = video_slots['val']
    val_dataset = CLEVRERSlotsVQADataset(**args)

    args['split'] = 'train'
    args['video_slots'] = video_slots['train']
    train_dataset = CLEVRERSlotsVQADataset(**args)

    return train_dataset, val_dataset
