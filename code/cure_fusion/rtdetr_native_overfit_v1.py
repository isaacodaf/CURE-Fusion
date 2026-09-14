"""RT-DETR fitting gates reuse frozen native fitting metrics and frame schedule."""
import json
from pathlib import Path
from cure_fusion.rtdetr_native_admission_v1 import sha256_file
from cure_fusion.dfine_full_native_overfit_v1 import (
    BUDGETS, batch_schedule, equal_tree, evaluate_mode,
    native_selected_records, state_digest)


def verify_admission(root, spec, scientific):
    if not isinstance(spec, dict):
        raise ValueError('A passed actual all128 RT-DETR admission must be bound before fitting')
    root = Path(root)
    for name in ('REPORT.json', 'ARTIFACT_HASHES.json', 'protocol.json'):
        if sha256_file(root/name) != spec['files'][name]:
            raise ValueError('Actual RT-DETR admission identity differs: '+name)
    local = Path(__file__).resolve().parents[2]
    peer_path = local/spec['independent_review_path']
    if sha256_file(peer_path) != spec['independent_review_sha256']:
        raise ValueError('Actual independent admission review differs')
    peer = json.loads(peer_path.read_text())
    if (peer['status'] != 'passed_rtdetr_native_four_class_actual_review_only' or peer['findings'] or
            peer['report_sha256'] != spec['files']['REPORT.json'] or
            peer['artifact_sha256'] != spec['files']['ARTIFACT_HASHES.json'] or
            peer['protocol_sha256'] != spec['files']['protocol.json']):
        raise ValueError('Review does not admit this actual RT-DETR execution')
    report = json.loads((root/'REPORT.json').read_text())
    parent = json.loads((root/'protocol.json').read_text())
    manifest = json.loads((root/'ARTIFACT_HASHES.json').read_text())
    for name in ('REPORT.json', 'protocol.json'):
        if manifest[name]['sha256'] != spec['files'][name] or manifest[name]['bytes'] != (root/name).stat().st_size:
            raise ValueError('Actual report/protocol artifact join differs')
    if (report['status'] != 'passed_rtdetr_native_four_class_one_step_admission_only' or
            report['optimizer_steps'] != 1 or report['CUDA_executed'] is not True or report['AP_evaluated'] is not False or
            report['protocol_sha256'] != spec['files']['protocol.json'] or
            parent['launch_authorized'] is not True or
            parent['prepared_protocol_sha256'] != scientific['admission_prepared_sha256']):
        raise ValueError('Actual successful V2 native admission required')
    for key in ('checkpoint_sha256','records_sha256','input_manifest_sha256','source_commit','config','seed',
                'native_overrides','deterministic_transform_ops','clip_max_norm','runtime_packages'):
        if parent[key] != scientific[key]:
            raise ValueError('Native scientific admission identity differs: '+key)
    if report['bound_files'] != parent['files'] or report['source_before'] != report['source_after']:
        raise ValueError('Actual source preservation differs')
    inv = json.loads((local/scientific['source_inventory_path']).read_text())
    expected = dict(commit=inv['commit'],archive_sha256=inv['archive_sha256'],files=inv['files'])
    if report['source_before'] != expected:
        raise ValueError('Actual author source identity differs')
    decode = report['one_step_native_decode']
    if (decode['images'] != 128 or decode['batches'] != 16 or decode['batch_size'] != 8 or
            decode['all_ids_exact'] is not True or decode['raw_and_selected_positive_geometry'] is not True or
            decode['exact_original_postprocessor'] is not True or decode['AP_evaluated'] is not False):
        raise ValueError('Complete actual128 native decode required')
    if report['selected_ids'] != scientific['sample_ids']:
        raise ValueError('Actual admitted population differs')
    mixed, empty = report['mixed_supervised_backward'], report['empty_only_backward']
    if len(mixed['components']) != 39 or len(mixed['raw_loss_calls']) != 26 or len(empty['components']) != 21 or len(empty['raw_loss_calls']) != 14:
        raise ValueError('Actual native criterion coverage differs')
    if any(v['class_abs'] <= 0 or v['box_abs'] <= 0 or v['objects'] <= 0 for v in mixed['positive_output_gradients'].values()):
        raise ValueError('Four-class matched connectivity missing')
    if set(mixed['positive_output_gradients']) != {'0','1','2','3'}:
        raise ValueError('Four-class connectivity axis differs')
    if empty['box_gradient_present_zero'] is not True or empty['parameters_unchanged'] is not True or empty['optimizer_steps'] != 0:
        raise ValueError('Actual empty backward contract failed')
    return dict(files=spec['files'], status=report['status'], initial_state=report['transfer']['loaded_state'],
                optimizer_groups=report['optimizer_groups'], optimizer_steps=1)
