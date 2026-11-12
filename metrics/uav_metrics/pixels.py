"""Bounded local pixel evidence. No visibility probability or semantic labels."""
from collections import Counter, defaultdict
from contextlib import contextmanager
import hashlib
import heapq
import io
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, uniform_filter
from metrics.uav_metrics.research import VERSION, bounded, finite, distribution, object_records


def pixel_features(gray):
    """All intensities normalized to [0,1]; tiny regions keep undefined values."""
    a = np.asarray(gray, dtype=float)/255
    if not a.size:
        return {}
    p = np.bincount(np.clip(np.rint(a*255), 0, 255).astype(np.uint8).ravel(), minlength=256)/a.size
    entropy = float(-(p[p > 0]*np.log2(p[p > 0])).sum())
    if min(a.shape) < 3:
        return {'mean_luminance': float(a.mean()), 'entropy_bits': entropy, 'contrast_std': float(a.std()),
                'edge_density': None, 'laplacian_variance': None, 'high_frequency_residual_mad': None,
                'local_contrast_rms': None, 'underexposed_fraction': float((a <= 5/255).mean()),
                'saturated_fraction': float((a >= 250/255).mean())}
    gy, gx = np.gradient(a)
    lap = -4*a[1:-1, 1:-1]+a[:-2, 1:-1]+a[2:, 1:-1]+a[1:-1, :-2]+a[1:-1, 2:]
    residual = a-gaussian_filter(a, 1, mode='reflect')
    local_mean = uniform_filter(a, size=3, mode='reflect')
    local_variance = np.maximum(0, uniform_filter(a*a, size=3, mode='reflect')-local_mean**2)
    return {'mean_luminance': float(a.mean()), 'entropy_bits': entropy, 'contrast_std': float(a.std()),
            'edge_density': float((np.hypot(gx, gy) > .08).mean()), 'laplacian_variance': float(lap.var()),
            'high_frequency_residual_mad': float(np.median(np.abs(residual-np.median(residual)))/.67448975),
            'local_contrast_rms': float(np.sqrt(local_variance).mean()),
            'underexposed_fraction': float((a <= 5/255).mean()), 'saturated_fraction': float((a >= 250/255).mean())}


def settings(g):
    spec = g.uav_parameters.get('local_analysis', {})
    if not isinstance(spec, dict) or set(spec)-{'max_images', 'max_objects', 'max_side', 'seed', 'ring_expansion', 'minimum_ring_pixels', 'max_neighbor_pairs'}:
        raise ValueError('Unknown local_analysis settings')
    result = {key: bounded(spec.get(key, default), lo, hi, key) for key, default, lo, hi in (
        ('max_images', 128, 1, 512), ('max_objects', 1024, 1, 4096), ('max_side', 512, 32, 1024),
        ('seed', 42, 0, 2**31-1), ('minimum_ring_pixels', 16, 4, 4096), ('max_neighbor_pairs', 100000, 1, 1000000))}
    expansion = spec.get('ring_expansion', 2.)
    if not finite(expansion) or not 1.1 <= expansion <= 5:
        raise ValueError('ring_expansion must be finite in [1.1,5]')
    return {**result, 'ring_expansion': expansion}


@contextmanager
def source_reader(ds):
    if getattr(ds, 'image_reader', None):
        with ds.image_reader() as archive:
            yield archive.read
    elif ds.source_path:
        from backend.dataset_archives import open_archive
        from backend.image_ingestion import IMAGE_SUFFIXES
        if Path(ds.name).suffix.lower() in IMAGE_SUFFIXES:
            yield lambda name: Path(ds.source_path).read_bytes()
        else:
            with open_archive(ds.source_path, ds.name) as archive:
                yield archive.read
    else:
        yield None


def crop_box(obj, width, height, expansion=1):
    x, y, w, h = obj['bbox']
    return (max(0, int(np.floor((x-expansion*w/2)*width))), max(0, int(np.floor((y-expansion*h/2)*height))),
            min(width, int(np.ceil((x+expansion*w/2)*width))), min(height, int(np.ceil((y+expansion*h/2)*height))))


def local_measurements(ds, g):
    spec = settings(g)
    from metrics.uav_metrics.vision import _job
    job = _job.get()
    key = ('local-pixels', ds.sha256, json.dumps(spec, sort_keys=True), json.dumps(g.uav_parameters.get('object_size_bins'), sort_keys=True))
    cache = job.setdefault('research', {}) if job is not None else {}
    if key in cache:
        return cache[key]
    objects = object_records(ds, g)
    by_image = defaultdict(list)
    for obj in objects:
        by_image[obj['image']].append(obj)
    rows = ds.frame[ds.frame.readable].to_dict('records')
    rank = lambda name: hashlib.sha256(f'{spec["seed"]}:{name}'.encode()).digest()
    selected = heapq.nsmallest(spec['max_images'], rows, key=lambda row: rank(row['image']))
    selected_names = {row['image'] for row in selected}
    chosen = heapq.nsmallest(spec['max_objects'], (o for o in objects if o['image'] in selected_names), key=lambda o: rank(f'{o["image"]}:{o["object_index"]}'))
    chosen_ids = {o['object_index'] for o in chosen}
    records, failures, inspected_images = [], [], []
    neighbor_pairs = 0
    try:
        with source_reader(ds) as read:
            if read:
                for row in selected:
                    image_objects = [o for o in by_image[row['image']] if o['object_index'] in chosen_ids]
                    if not image_objects:
                        continue
                    try:
                        raw = read(row['image'])
                        if len(raw) > 20*1024**2 or hashlib.sha256(raw).hexdigest() != ds.image_hashes.get(row['image']):
                            raise ValueError('Image bytes changed or exceed decode budget')
                        with Image.open(io.BytesIO(raw)) as im:
                            if im.width*im.height > 20_000_000 or getattr(im, 'n_frames', 1) != 1:
                                raise ValueError('Unsafe pixel dimensions/frame count')
                            if (im.width, im.height) != (row['width'], row['height']):
                                raise ValueError('Image dimensions changed')
                            gray = im.convert('L')
                            gray.thumbnail((spec['max_side'], spec['max_side']))
                            a = np.asarray(gray, dtype=np.uint8)
                        height, width = a.shape
                        inspected_images.append(row['image'])
                        for obj in image_objects:
                            left, top, right, bottom = crop_box(obj, width, height)
                            xl, yt, xr, yb = crop_box(obj, width, height, spec['ring_expansion'])
                            target = a[top:bottom, left:right]
                            area = a[yt:yb, xl:xr]
                            mask = np.ones(area.shape, dtype=bool)
                            mask[top-yt:bottom-yt, left-xl:right-xl] = False
                            nearby = overlaps = 0
                            neighborhood_complete = True
                            for other in by_image[row['image']]:
                                if other['object_index'] == obj['object_index']:
                                    continue
                                if neighbor_pairs >= spec['max_neighbor_pairs']:
                                    neighborhood_complete = False
                                    break
                                neighbor_pairs += 1
                                ol, ot, oright, ob = crop_box(other, width, height)
                                if ol < xr and oright > xl and ot < yb and ob > yt:
                                    nearby += 1
                                    mask[max(0, ot-yt):min(yb-yt, ob-yt), max(0, ol-xl):min(xr-xl, oright-xl)] = False
                                if ol < right and oright > left and ot < bottom and ob > top:
                                    overlaps += 1
                            object_features = pixel_features(target)
                            ring = area[mask]
                            ring_features = {}
                            if len(ring) >= spec['minimum_ring_pixels'] and min(area.shape) >= 3 and neighborhood_complete:
                                p = np.bincount(ring, minlength=256)/len(ring)
                                gy, gx = np.gradient(area.astype(float)/255)
                                ring_features = {'entropy_bits': float(-(p[p > 0]*np.log2(p[p > 0])).sum()),
                                                 'contrast_std': float(ring.std()/255), 'mean_luminance': float(ring.mean()/255),
                                                 'edge_density': float((np.hypot(gx, gy)[mask] > .08).mean())}
                            background_mean = ring_features.get('mean_luminance')
                            foreground_mean = object_features.get('mean_luminance')
                            difference = abs(foreground_mean-background_mean) if background_mean is not None and foreground_mean is not None else None
                            normalized = difference/max(1/255, foreground_mean+background_mean) if difference is not None else None
                            separation = object_features.get('edge_density')-ring_features['edge_density'] if ring_features and object_features.get('edge_density') is not None else None
                            records.append({'image': row['image'], 'object_index': obj['object_index'], 'class': obj['class'], 'size_bin': obj['size_bin'],
                                            'native_area_pixels': obj['area_pixels'], 'object': object_features, 'background_ring': ring_features,
                                            'ring_pixels': len(ring), 'ring_status': 'measured' if ring_features else 'insufficient_evidence',
                                            'nearby_annotated_objects': nearby if neighborhood_complete else None, 'overlapping_boxes': overlaps if neighborhood_complete else None,
                                            'luminance_difference': difference, 'normalized_local_contrast': normalized, 'edge_separation': separation})
                    except (OSError, ValueError, KeyError) as exc:
                        failures.append({'image': row['image'], 'reason': str(exc)})
            else:
                failures.append({'reason': 'Staged image bytes unavailable; no local measurement invented.'})
    except (OSError, ValueError, KeyError) as exc:
        failures.append({'reason': str(exc)})
    keys = [('object', 'laplacian_variance'), ('object', 'high_frequency_residual_mad'), ('object', 'local_contrast_rms'),
            ('object', 'underexposed_fraction'), ('object', 'saturated_fraction'), ('background_ring', 'entropy_bits'),
            ('background_ring', 'edge_density'), ('background_ring', 'contrast_std')]
    def summarize(items):
        units={'entropy_bits':'bits','edge_density':'fraction','laplacian_variance':'normalized intensity squared',
               'high_frequency_residual_mad':'normalized intensity','local_contrast_rms':'normalized intensity',
               'contrast_std':'normalized intensity','underexposed_fraction':'fraction','saturated_fraction':'fraction'}
        result = {f'{region}.{field}': distribution([r[region].get(field) for r in items], units[field]) for region, field in keys}
        for field in ('luminance_difference', 'normalized_local_contrast', 'edge_separation', 'nearby_annotated_objects', 'overlapping_boxes'):
            result[field] = distribution([r.get(field) for r in items], 'objects' if field in ('nearby_annotated_objects', 'overlapping_boxes') else 'normalized intensity' if field=='luminance_difference' else 'difference in fractions' if field=='edge_separation' else 'ratio')
        return result
    # Relative ranks identify inspection candidates, never verified difficult cases.
    candidates = sorted((r for r in records if r['background_ring']), key=lambda r: (-r['background_ring']['entropy_bits'], r['image'], r['object_index']))[:20]
    cross = defaultdict(Counter)
    entropy_values = [r['background_ring']['entropy_bits'] for r in records if r['background_ring']]
    edges = list(map(float, np.quantile(entropy_values, [.25, .75]))) if entropy_values else None
    for r in records:
        entropy = r['background_ring'].get('entropy_bits')
        label = 'unknown_ring' if entropy is None else 'lower_sample_quartile' if entropy < edges[0] else 'upper_sample_quartile' if entropy > edges[1] else 'central_sample_range'
        cross[r['size_bin']][label] += 1
    output = {'version': VERSION, 'status': 'measured' if records else 'insufficient_evidence', 'evidence_type': 'observed_pixel_proxies',
              'sampling': {'strategy': 'SHA256(seed:image) bottom-k images, then SHA256(seed:image:object-index) bottom-k objects',
                           'seed': spec['seed'], 'settings': spec, 'image_population': len(rows), 'selected_images': len(selected),
                           'images_inspected': len(inspected_images), 'object_population': len(objects), 'objects_inspected': len(records),
                           'image_coverage': len(inspected_images)/len(rows) if rows else None, 'object_coverage': len(records)/len(objects) if objects else None,
                           'scope': 'full_valid_object_population' if len(records) == len(objects) and objects else 'bounded_sample',
                           'neighbor_comparisons': neighbor_pairs, 'extra_decode_reason': 'Object-local crops unavailable from existing ingestion-wide summaries; one additional decode per sampled image, shared across UAV-12/14.'},
              'distributions': summarize(records), 'per_class': {c: summarize([r for r in records if r['class'] == c]) for c in sorted({r['class'] for r in records})},
              'per_size': {b: summarize([r for r in records if r['size_bin'] == b]) for b in sorted({r['size_bin'] for r in records})},
              'size_local_complexity_cross_tab': {b: dict(c) for b, c in cross.items()},
              'complexity_cross_tab_definition': {'quantity': 'ring entropy bits', 'sample_q25_q75': edges, 'policy': 'Sample-relative bins; ties in central range; not calibrated clutter or difficulty categories'},
              'review_candidates': [{'image': r['image'], 'object_index': r['object_index'], 'class': r['class'], 'reason': 'Highest measured surrounding-ring entropy in inspected sample',
                                     'entropy_bits': r['background_ring']['entropy_bits'], 'status': 'unverified_candidate'} for r in candidates],
              'failures': failures[:20], 'limitations': ['Bounding boxes contain background; rings exclude known annotated boxes only.',
                    'Measurements use aspect-preserving bounded grayscale thumbnails; tiny crops have undefined detail proxies.',
                    'High-frequency residuals mix texture and noise; Laplacian variance mixes sharpness and noise.',
                    'Low brightness is not night, high entropy is not difficulty, contrast is not semantic visibility.',
                    'Candidate ranks are relative to the inspected sample; no detection probability or calibrated confidence interval.']}
    cache[key] = output
    return output
