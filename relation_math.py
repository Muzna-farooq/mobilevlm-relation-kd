"""Differentiable prompt-to-image cosine relations; no MobileVLM imports."""
import torch
import torch.nn.functional as F


def relation_positions(input_ids, raw_labels, expanded_length, image_token_index, ignore_index):
    """Batch size one; one image placeholder; exclude answer/EOS tokens."""
    ids, labels = input_ids[0], raw_labels[0]
    image = (ids == image_token_index).nonzero(as_tuple=True)[0]
    answer = (labels != ignore_index).nonzero(as_tuple=True)[0]
    if image.numel() != 1 or answer.numel() == 0:
        raise ValueError('One image and a nonempty supervised answer are required.')
    image_at, answer_at = int(image[0]), int(answer[0])
    n_visual = expanded_length - ids.numel() + 1
    if n_visual != 144 or answer_at <= image_at + 1:
        raise ValueError(f'Unexpected alignment: visual tokens={n_visual}.')
    visual = torch.arange(image_at, image_at + n_visual, device=ids.device)
    # Text following the image includes the question, instruction and assistant header.
    # This is a representation relation matrix, NOT an attention probability map.
    question = torch.arange(image_at + n_visual, answer_at + n_visual - 1, device=ids.device)
    return question, visual


def relation_matrix(hidden, question, visual):
    h = hidden[0].float()
    q = F.normalize(h.index_select(0, question.to(h.device)), dim=-1, eps=1e-6)
    v = F.normalize(h.index_select(0, visual.to(h.device)), dim=-1, eps=1e-6)
    return q @ v.T


def relation_loss(student_relation, teacher_relation):
    if student_relation.shape != teacher_relation.shape:
        raise ValueError('Teacher and student relation positions do not align.')
    return F.mse_loss(student_relation.float(), teacher_relation.to(student_relation.device).float())
