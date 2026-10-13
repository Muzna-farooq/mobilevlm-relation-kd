"""Muzna's Relation KD experiment, matched to the supplied MobileVLM baseline.

Stages: cache frozen teacher targets from TRAIN, train a fresh student, evaluate.
Requires the team's GQA Kaggle dataset, shared backup, Internet, and two T4 GPUs.
This implements a proposed cosine-relation objective, not an Align-KD replication.
"""
import argparse
import csv
import gc
import hashlib
import json
import math
import random
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from huggingface_hub import snapshot_download
from relation_math import relation_positions, relation_matrix, relation_loss


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding='utf-8')


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def normalize(text):
    text = re.sub(r'^[^\w]+|[^\w]+$', '', text.lower().strip())
    return re.sub(r'\s+', ' ', text)


def select_rows(rows, limit, seed):
    rows = sorted(rows, key=lambda r: r['id'])
    random.Random(seed).shuffle(rows)
    return rows if limit is None else rows[:limit]


def setup(args):
    shared = Path(args.shared_dir)
    original = json.loads((shared / 'shared_config.json').read_text())
    lock = json.loads((shared / 'versions_lock.json').read_text())
    if original['versions'] != lock:
        raise ValueError('Shared model revisions disagree.')
    cfg = dict(original)
    cfg.update(data_dir=str(Path(args.data_dir)), output_dir=str(Path(args.output_dir)),
               profile=args.profile,
               train_limit=128 if args.profile == 'smoke' else original['train_limit'],
               validation_limit=20 if args.profile == 'smoke' else original['validation_limit'])
    # Preserve the shared dev split, even in smoke mode: no dev_mini substitution.
    out = Path(cfg['output_dir']); out.mkdir(parents=True, exist_ok=True)
    root = Path(cfg['data_dir'])
    for name, expected in original['dataset_hashes'].items():
        if sha(root / (name + '.json')) != expected:
            raise ValueError(f'Dataset hash mismatch: {name}')
    all_train = json.loads((root / (cfg['train_file'] + '.json')).read_text())
    all_dev = json.loads((root / (cfg['validation_file'] + '.json')).read_text())
    test = json.loads((root / 'testdev.json').read_text())
    for rows in (all_train, all_dev, test):
        if len({r['id'] for r in rows}) != len(rows):
            raise ValueError('Duplicate question IDs.')
    groups = [{r['image'] for r in rows} for rows in (all_train, all_dev, test)]
    if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
        raise ValueError('Train/dev/test image overlap.')
    # Check the team's original full subset order before applying smoke limits.
    for name, rows in [('train', all_train), ('validation', all_dev)]:
        complete = select_rows(rows, original[name + '_limit'], cfg['seed'])
        ids = '\n'.join(r['id'] for r in complete)
        if hashlib.sha256(ids.encode()).hexdigest() != original['selected_id_hashes'][name]:
            raise ValueError(f'Shared selection hash mismatch: {name}')
    train = select_rows(all_train, cfg['train_limit'], cfg['seed'])
    dev = select_rows(all_dev, cfg['validation_limit'], cfg['seed'])
    for rows in (train, dev):
        if not rows or not all((root / r['image']).is_file() for r in rows):
            raise ValueError('Empty split or missing images.')
    cfg['selected_id_hashes'] = {}
    for name, rows in [('train', train), ('validation', dev)]:
        (out / (name + '_question_ids.txt')).write_text('\n'.join(r['id'] for r in rows))
        cfg['selected_id_hashes'][name] = sha(out / (name + '_question_ids.txt'))
    method = {'name': 'prompt_image_cosine_relation_kd', 'lambda_relation': args.lambda_relation,
              'layer': 'final_decoder_post_norm', 'visual_tokens': 144,
              'text_positions': 'all post-image prompt tokens; excludes answer and EOS',
              'loss': 'answer CE + lambda_relation * mean squared cosine-relation difference',
              'cache_precision': 'float16', 'epochs': cfg['epochs']}
    if cfg['epochs'] != 1 or cfg['micro_batch_size'] != 1:
        raise ValueError('This implementation matches the one-epoch, batch-size-one baseline.')
    for name, obj in [('shared_config.json', cfg), ('method_config.json', method), ('versions_lock.json', lock)]:
        target = out / name
        if target.exists() and json.loads(target.read_text()) != obj:
            raise ValueError(f'{name} changed. Use a new output folder.')
        write_json(target, obj)
    if not torch.cuda.is_available():
        raise RuntimeError('Enable GPU in Kaggle notebook settings.')
    import transformers, accelerate, peft, tokenizers
    versions = {'transformers': transformers.__version__, 'accelerate': accelerate.__version__,
                'peft': peft.__version__, 'tokenizers': tokenizers.__version__}
    expected = {'transformers': '4.37.2', 'accelerate': '0.27.2', 'peft': '0.7.1', 'tokenizers': '0.15.2'}
    if versions != expected:
        raise RuntimeError(f'Install pinned dependencies and restart kernel. Found: {versions}')
    write_json(out / 'environment.json', dict(versions, torch=torch.__version__,
        GPUs=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]))
    repo = Path(args.repo_dir)
    if not repo.exists():
        subprocess.check_call(['git', 'clone', 'https://github.com/Meituan-AutoML/MobileVLM.git', str(repo)])
    subprocess.check_call(['git', '-C', str(repo), 'checkout', lock['source_commit']])
    sys.path.insert(0, str(repo))
    from mobilevlm.model.mobilellama import MobileVLMConfig, MobileLlamaForCausalLM
    print(f"Profile={args.profile}; train={len(train)}; validation={len(dev)}", flush=True)
    return cfg, lock, train, dev


def snapshot(cfg, lock, teacher=False):
    vision = snapshot_download('openai/clip-vit-large-patch14-336', revision=lock['vision_revision'],
                               ignore_patterns=['*.h5', '*.msgpack', '*.onnx'])
    kind = 'teacher' if teacher else 'student'
    source = Path(snapshot_download(cfg[kind + '_checkpoint'], revision=lock[kind + '_revision'],
                                   ignore_patterns=['*.h5', '*.msgpack', '*.onnx']))
    overlay = Path(cfg['output_dir']) / (kind + '_snapshot')
    overlay.mkdir(exist_ok=True)
    for f in source.iterdir():
        dst = overlay / f.name
        if f.name != 'config.json' and not dst.exists():
            dst.symlink_to(f)
    config = json.loads((source / 'config.json').read_text())
    config['mm_vision_tower'] = vision
    write_json(overlay / 'config.json', config)
    return overlay


def load_model(path, teacher=False):
    from transformers import AutoConfig
    from mobilevlm.model.mobilevlm import load_pretrained_model
    device_map = {'': 0}
    if teacher:
        if torch.cuda.device_count() < 2:
            raise RuntimeError('Use 2 x T4 GPUs for the shared FP16 7B teacher.')
        config = AutoConfig.from_pretrained(str(path))
        device_map = {'model.embed_tokens': 0, 'model.vision_tower': 0,
                      'model.mm_projector': 0, 'model.norm': 1, 'lm_head': 1}
        cut = config.num_hidden_layers * 3 // 8
        device_map.update({f'model.layers.{i}': 0 if i < cut else 1
                          for i in range(config.num_hidden_layers)})
    tokenizer, model, processor, _ = load_pretrained_model(str(path), load_8bit=False,
        load_4bit=False, device_map=device_map, device='cuda:0')
    if any(p.is_meta for p in model.parameters()):
        raise RuntimeError('Some model weights were not loaded.')
    if model.config.mm_projector_type != 'ldpnetv2':
        raise ValueError('Expected the shared LDPNetV2 architecture.')
    model.config.image_aspect_ratio = 'pad'
    model.requires_grad_(False); model.eval()
    return tokenizer, model, processor


def encode(row, tokenizer, processor, base, cfg, training):
    from mobilevlm.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
    from mobilevlm.conversation import conv_templates
    from mobilevlm.utils import process_images, tokenizer_image_token
    human = row['conversations'][0]['value']
    if human.count('<image>') != 1 or not human.endswith(cfg['answer_suffix']):
        raise ValueError('Dataset prompt differs from the shared baseline.')
    conv = conv_templates[cfg['conversation_template']].copy()
    conv.append_message(conv.roles[0], human); conv.append_message(conv.roles[1], None)
    ids = tokenizer_image_token(conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
    labels = None
    if training:
        answer = tokenizer(' ' + row['conversations'][1]['value'], add_special_tokens=False).input_ids
        answer = torch.tensor(answer + [tokenizer.eos_token_id])
        labels = torch.cat([torch.full_like(ids, IGNORE_INDEX), answer])
        ids = torch.cat([ids, answer])
    if ids.numel() + 143 + (0 if training else cfg['max_new_tokens']) > cfg['max_sequence_length']:
        raise ValueError(f"Sequence exceeds configured length: {row['id']}")
    with Image.open(Path(cfg['data_dir']) / row['image']) as image:
        pixels = process_images([image.convert('RGB')], processor, base.config)
    if not isinstance(pixels, torch.Tensor):
        raise ValueError('Expected a tensor from the shared image processor.')
    device = base.get_input_embeddings().weight.device
    return dict(input_ids=ids[None].to(device), attention_mask=torch.ones(1, ids.numel(), dtype=torch.bool, device=device),
                labels=None if labels is None else labels[None].to(device), images=pixels.to(device, torch.float16))


def decode(base, batch):
    ids, mask, _, embeds, labels = base.prepare_inputs_labels_for_multimodal(
        batch['input_ids'], batch['attention_mask'], None, batch['labels'], batch['images'])
    decoded = base.get_model()(input_ids=ids, attention_mask=mask, inputs_embeds=embeds,
        use_cache=False, output_hidden_states=False, return_dict=True)
    # Only the final hidden states are used; no all-layer attention allocation.
    return decoded.last_hidden_state, labels


def compute_relations(hidden, batch):
    from mobilevlm.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
    q, v = relation_positions(batch['input_ids'], batch['labels'], hidden.shape[1], IMAGE_TOKEN_INDEX, IGNORE_INDEX)
    return relation_matrix(hidden, q, v)


def torch_load(path):
    # Only load local targets/checkpoints made by this script (not arbitrary uploads).
    return torch.load(path, map_location='cpu', weights_only=False)


def save_atomic(obj, path):
    path = Path(path); tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(obj, tmp); tmp.replace(path)


def build_cache(cfg, lock, train):
    cache = Path(cfg['output_dir']) / 'teacher_relation_cache'; cache.mkdir(exist_ok=True)
    contract = {'teacher_revision': lock['teacher_revision'], 'source_commit': lock['source_commit'],
                'vision_revision': lock['vision_revision'], 'train_ids_sha256': sha(Path(cfg['output_dir']) / 'train_question_ids.txt'),
                'dataset_hash': cfg['dataset_hashes'][cfg['train_file']], 'method': 'post_image_prompt_to_visual_cosine_final_layer'}
    path = cache / 'contract.json'
    if path.exists() and json.loads(path.read_text()) != contract:
        raise ValueError('Cache contract mismatch.')
    write_json(path, contract)
    pending = [r for r in train if not (cache / (r['id'] + '.pt')).exists()]
    if not pending:
        print('Teacher cache is complete. No model download required.'); return
    tokenizer, model, processor = load_model(snapshot(cfg, lock, teacher=True), teacher=True)
    write_json(cache / 'processor.json', processor.to_dict())
    tokenizer.save_pretrained(cache / 'tokenizer')
    write_json(cache / 'architecture.json', {'layers': model.config.num_hidden_layers, 'hidden_size': model.config.hidden_size})
    from tqdm.auto import tqdm
    with torch.inference_mode():
        for row in tqdm(pending, desc='Cache TRAIN teacher relations'):
            batch = encode(row, tokenizer, processor, model, cfg, training=True)
            hidden, labels = decode(model, batch)
            target = compute_relations(hidden, batch)
            save_atomic({'id': row['id'], 'image': row['image'], 'input_ids': batch['input_ids'].cpu(),
                         'aligned_labels': labels.cpu(), 'relations': target.cpu().half()}, cache / (row['id'] + '.pt'))
            del batch, hidden, labels, target
    del tokenizer, model, processor; gc.collect(); torch.cuda.empty_cache()
    print('Teacher TRAIN cache complete; teacher released. Validation predictions are not KD targets.')


def make_student(base, cfg):
    from peft import LoraConfig, get_peft_model
    targets = [name for name, mod in base.named_modules() if isinstance(mod, torch.nn.Linear)
               and name.startswith('model.layers.') and name.rsplit('.', 1)[-1] in cfg['lora_targets']]
    if not targets: raise ValueError('No language decoder LoRA targets found.')
    base.config.use_cache = False
    model = get_peft_model(base, LoraConfig(r=cfg['lora_rank'], lora_alpha=cfg['lora_alpha'],
        lora_dropout=cfg['lora_dropout'], target_modules=targets, bias='none', task_type='CAUSAL_LM'))
    for p in model.parameters():
        if p.requires_grad: p.data = p.data.float()
    # Explicit non-reentrant checkpointing avoids the warning in the supplied notebook.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    return model


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all()}


def restore_rng(rng):
    random.setstate(rng['python']); np.random.set_state(rng['numpy']); torch.set_rng_state(rng['torch'])
    torch.cuda.set_rng_state_all(rng['cuda'])


def train_student(cfg, lock, train, args):
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
    from mobilevlm.constants import IGNORE_INDEX
    from tqdm.auto import tqdm
    out = Path(cfg['output_dir']); cache = out / 'teacher_relation_cache'
    if not all((cache / (r['id'] + '.pt')).exists() for r in train):
        raise RuntimeError('Run the cache stage first; teacher predictions alone are insufficient.')
    if (out / 'training_complete.json').exists():
        print('Training already complete. Use the evaluate stage.'); return
    seed_all(cfg['seed'])
    tokenizer, base, processor = load_model(snapshot(cfg, lock))
    teacher_tokenizer = AutoTokenizer.from_pretrained(str(cache / 'tokenizer'), use_fast=False)
    if tokenizer.get_vocab() != teacher_tokenizer.get_vocab():
        raise ValueError('Teacher/student vocabularies differ.')
    if json.loads(json.dumps(processor.to_dict())) != json.loads((cache / 'processor.json').read_text()):
        raise ValueError('Teacher/student image preprocessing differs.')
    del teacher_tokenizer
    model = make_student(base, cfg)
    write_json(out / 'student_architecture.json', {'layers': base.config.num_hidden_layers,
        'hidden_size': base.config.hidden_size, 'parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)})
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    accumulation = cfg['gradient_accumulation']; steps = math.ceil(len(train) / accumulation)
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(steps * cfg['warmup_ratio']), steps)
    scaler = torch.cuda.amp.GradScaler()
    order = list(train); random.Random(cfg['seed']).shuffle(order)
    start_index = updates = 0; sums = [0.0, 0.0, 0.0]
    resume = out / 'resume.pt'
    if resume.exists():
        if not args.resume: raise RuntimeError('Partial checkpoint exists. Add --resume to continue it.')
        saved = torch_load(resume)
        set_peft_model_state_dict(model, saved['adapters'])
        optimizer.load_state_dict(saved['optimizer']); scheduler.load_state_dict(saved['scheduler'])
        scaler.load_state_dict(saved['scaler']); restore_rng(saved['rng'])
        start_index, updates, sums = saved['next_index'], saved['updates'], saved['sums']
        print(f'Resuming at example {start_index}; updates={updates}', flush=True)
    elif args.resume: raise RuntimeError('No resume.pt checkpoint found.')
    model.train(); base.get_vision_tower().eval(); base.get_model().mm_projector.eval()
    optimizer.zero_grad(set_to_none=True)
    def checkpoint(next_index):
        save_atomic({'adapters': {k: v.detach().cpu() for k, v in get_peft_model_state_dict(model).items()},
            'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
            'rng': rng_state(), 'next_index': next_index, 'updates': updates, 'sums': sums}, resume)
    for index in tqdm(range(start_index, len(order)), desc='Relation KD epoch 1'):
        row = order[index]
        batch = encode(row, tokenizer, processor, base, cfg, training=True)
        cached = torch_load(cache / (row['id'] + '.pt'))
        if cached['id'] != row['id'] or cached['image'] != row['image'] or not torch.equal(cached['input_ids'], batch['input_ids'].cpu()):
            raise ValueError(f"Teacher/student sequence mismatch: {row['id']}")
        with torch.autocast('cuda', dtype=torch.float16):
            hidden, labels = decode(base, batch)
            if not torch.equal(cached['aligned_labels'], labels.cpu()):
                raise ValueError('Expanded answer positions differ.')
            logits = base.lm_head(hidden)
            gt = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                                 labels[:, 1:].reshape(-1).to(logits.device), ignore_index=IGNORE_INDEX)
            rel = relation_loss(compute_relations(hidden, batch), cached['relations'])
            loss = gt + args.lambda_relation * rel
        if not torch.isfinite(loss): raise RuntimeError(f"Nonfinite loss at {row['id']}")
        if index == start_index:
            # Verify KD itself reaches trainable adapters, independently of the GT loss.
            if not rel.requires_grad:
                raise RuntimeError('Relation loss was detached from the student graph.')
            grads = torch.autograd.grad(rel, params, retain_graph=True, allow_unused=True)
            kd_gradient_norm = sum(float(g.detach().abs().sum()) for g in grads if g is not None)
            del grads
            if not math.isfinite(kd_gradient_norm) or kd_gradient_norm <= 0:
                raise RuntimeError('No finite nonzero relation gradient reaches the LoRA adapters.')
            print(f'Relation gradient check passed: L1 norm={kd_gradient_norm:.6g}', flush=True)
        window = min(accumulation, len(order) - (index // accumulation) * accumulation)
        scaler.scale(loss / window).backward()
        for j, value in enumerate((loss, gt, rel)): sums[j] += float(value.detach())
        del hidden, labels, logits, gt, rel, loss, batch, cached
        if (index + 1) % accumulation == 0 or index + 1 == len(order):
            scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(params, cfg['max_grad_norm'])
            scale = scaler.get_scale(); scaler.step(optimizer); scaler.update()
            if scaler.get_scale() >= scale: scheduler.step(); updates += 1
            optimizer.zero_grad(set_to_none=True)
            if updates % cfg.get('checkpoint_every_updates', 200) == 0 or index + 1 == len(order):
                checkpoint(index + 1)
                print(f'processed={index+1}; updates={updates}; CE={sums[1]/(index+1):.5f}; relation={sums[2]/(index+1):.5f}', flush=True)
    adapter = out / 'relation_adapter_epoch_1'; model.save_pretrained(adapter); tokenizer.save_pretrained(adapter)
    write_json(out / 'training_complete.json', {'examples': len(order), 'optimizer_updates': updates,
        'total_loss_mean': sums[0]/len(order), 'gt_loss_mean': sums[1]/len(order),
        'relation_loss_mean': sums[2]/len(order), 'lambda_relation': args.lambda_relation})
    print('Training complete. Saved LoRA adapters, not a standalone 3B model.', flush=True)


def evaluate(cfg, lock, dev, args):
    from peft import PeftModel
    from tqdm.auto import tqdm
    out = Path(cfg['output_dir']); adapter = out / 'relation_adapter_epoch_1'
    if not (out / 'training_complete.json').exists(): raise RuntimeError('Complete training first.')
    seed_all(cfg['seed'])
    tokenizer, base, processor = load_model(snapshot(cfg, lock))
    model = PeftModel.from_pretrained(base, str(adapter), is_trainable=False)
    model.eval(); model.config.use_cache = True
    details = []; totals, correct = Counter(), Counter()
    # Warm-up is unscored; all recorded timings synchronize every GPU.
    for index, row in enumerate(tqdm([dev[0]] + dev, desc='Relation KD validation')):
        batch = encode(row, tokenizer, processor, base, cfg, training=False)
        for g in range(torch.cuda.device_count()):
            torch.cuda.synchronize(g); torch.cuda.reset_peak_memory_stats(g)
        start = time.perf_counter()
        with torch.inference_mode():
            result = model.generate(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                images=batch['images'], do_sample=False, num_beams=1, max_new_tokens=cfg['max_new_tokens'],
                use_cache=True, pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)
        for g in range(torch.cuda.device_count()): torch.cuda.synchronize(g)
        elapsed = time.perf_counter() - start; n = batch['input_ids'].shape[1]
        if not torch.equal(result[0, :n], batch['input_ids'][0]): raise ValueError('Unexpected generation prefix.')
        prediction = tokenizer.decode(result[0, n:], skip_special_tokens=True).strip()
        if index == 0: continue
        gold = row['conversations'][1]['value']
        ok = bool(normalize(prediction)) and normalize(prediction) == normalize(gold)
        for group in ['all', 'semantic:' + row['meta']['semantic'], 'structural:' + row['meta']['structural']]:
            totals[group] += 1; correct[group] += int(ok)
        generated = result.shape[1] - n
        details.append({'question_id': row['id'], 'image_id': Path(row['image']).stem,
            'semantic_type': row['meta']['semantic'], 'structural_type': row['meta']['structural'],
            'prediction': prediction, 'ground_truth': gold, 'correct': int(ok), 'latency_ms': elapsed * 1000,
            'generated_tokens': generated, 'tokens_per_s': generated / elapsed,
            'peak_gpu_memory_mb': sum(torch.cuda.max_memory_allocated(g) for g in range(torch.cuda.device_count())) / 1e6})
    metrics = {'n': len(dev), 'accuracy': {k: correct[k]/totals[k] for k in totals},
        'latency_ms_mean': float(np.mean([r['latency_ms'] for r in details])),
        'tokens_per_s_mean': float(np.mean([r['tokens_per_s'] for r in details])),
        'peak_gpu_memory_mb': max(r['peak_gpu_memory_mb'] for r in details),
        'parameters': sum(p.numel() for p in model.parameters()), 'model_name': 'student_3b_relation_kd',
        'split': cfg['validation_file'], 'profile': cfg['profile'], 'timing_note': 'LoRA unmerged; generation only, excluding image preprocessing'}
    write_json(out / 'relation_kd_validation_metrics.json', metrics)
    with (out / 'relation_kd_validation_predictions.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=details[0]); writer.writeheader(); writer.writerows(details)
    comparisons = []
    wanted = {r['id']: r for r in dev}
    for name, filename in [('teacher_7b', 'teacher_7b_dev_predictions.csv'),
                           ('student_3b_initial', 'student_3b_initial_dev_predictions.csv')]:
        file = Path(args.shared_dir) / filename
        if not file.exists(): continue
        with file.open() as f: references = {r['question_id']: r for r in csv.DictReader(f)}
        if not set(wanted) <= references.keys(): raise ValueError('Reference prediction IDs missing.')
        if any(normalize(references[k]['ground_truth']) != normalize(wanted[k]['conversations'][1]['value']) for k in wanted):
            raise ValueError('Reference ground-truth answers differ.')
        score = sum(bool(normalize(references[k]['prediction'])) and
                    normalize(references[k]['prediction']) == normalize(wanted[k]['conversations'][1]['value'])
                    for k in wanted) / len(wanted)
        relational = [k for k in wanted if wanted[k]['meta']['semantic'] == 'rel']
        rel_score = sum(normalize(references[k]['prediction']) == normalize(references[k]['ground_truth'])
                        for k in relational) / len(relational) if relational else None
        comparisons.append({'model': name, 'n': len(dev), 'accuracy_percent': score * 100,
                            'relational_accuracy_percent': None if rel_score is None else rel_score * 100})
    if args.gt_baseline_csv:
        with Path(args.gt_baseline_csv).open() as f:
            reference = {r['question_id']: r for r in csv.DictReader(f)}
        if not set(wanted) <= reference.keys():
            raise ValueError('GT baseline predictions must cover the same validation IDs.')
        if any(normalize(reference[k]['ground_truth']) != normalize(wanted[k]['conversations'][1]['value']) for k in wanted):
            raise ValueError('GT baseline ground-truth answers differ.')
        subset = [k for k in wanted if wanted[k]['meta']['semantic'] == 'rel']
        def baseline_score(keys):
            return sum(bool(normalize(reference[k]['prediction'])) and
                normalize(reference[k]['prediction']) == normalize(reference[k]['ground_truth'])
                for k in keys) / len(keys) * 100 if keys else None
        comparisons.append({'model': 'student_3b_gt_trained', 'n': len(dev),
                            'accuracy_percent': baseline_score(list(wanted)),
                            'relational_accuracy_percent': baseline_score(subset)})
    comparisons.append({'model': metrics['model_name'], 'n': len(dev),
        'accuracy_percent': metrics['accuracy']['all'] * 100,
        'relational_accuracy_percent': metrics['accuracy'].get('semantic:rel', float('nan')) * 100})
    teacher = next((r['accuracy_percent'] for r in comparisons if r['model'] == 'teacher_7b'), None)
    for row in comparisons:
        row['teacher_score_retention_percent'] = row['accuracy_percent'] / teacher * 100 if teacher else None
    with (out / 'comparison.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=comparisons[0]); writer.writeheader(); writer.writerows(comparisons)
    print(json.dumps(metrics, indent=2)); print(json.dumps(comparisons, indent=2))
    print('Add the separately GT-trained baseline on the SAME dev IDs before attributing gains to KD.')
    print('Validation results only. No testdev evaluation or fabricated KD scores.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['cache', 'train', 'evaluate'], required=True)
    p.add_argument('--profile', choices=['smoke', 'main'], default='smoke')
    p.add_argument('--data-dir', required=True); p.add_argument('--shared-dir', required=True)
    p.add_argument('--output-dir', required=True); p.add_argument('--repo-dir', default='/kaggle/working/MobileVLM')
    p.add_argument('--lambda-relation', type=float, default=1.0, help='Pilot weight, not a tuned optimum')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--gt-baseline-csv', help='Optional GT-trained baseline predictions from the team; same training profile required')
    args = p.parse_args()
    if not math.isfinite(args.lambda_relation) or args.lambda_relation < 0:
        p.error('--lambda-relation must be finite and nonnegative')
    cfg, lock, train, dev = setup(args)
    if args.stage == 'cache': build_cache(cfg, lock, train)
    elif args.stage == 'train': train_student(cfg, lock, train, args)
    else: evaluate(cfg, lock, dev, args)


if __name__ == '__main__':
    main()
