"""Enable the locally verified joint model. No upload or external deployment."""
from datetime import datetime, timedelta, timezone
import json
import torch
from core import HERE, digest, save_json


def main():
    folder = HERE / 'artifacts/generation/joint'
    verification = json.loads((folder / 'verification.json').read_text(encoding='utf-8'))
    report = json.loads((folder / 'metrics.json').read_text(encoding='utf-8'))
    state = torch.load(folder / 'model.pt', map_location='cpu', weights_only=True)
    assert verification['model_sha256'] == digest(folder / 'model.pt')
    assert verification['manifest_sha256'] == digest(folder / 'manifest.csv') == state['manifest_sha256']
    assert verification['checks']['no_face_is_not_real_label'] == 'passed'
    assert state['threshold_strategy'] == 'Midpoint of optimal validation balanced-accuracy interval'
    # Model choice uses validation, never whether the user's one image passes.
    if report['splits']['val']['balanced_accuracy'] < .9:
        raise ValueError('联合模型未达到预设验证集门槛，不切换默认版本')
    save_json(folder / 'READY.json', dict(model_sha256=verification['model_sha256'],
        activated_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
        criterion='Validation balanced accuracy >= 0.90 and functional parity checks passed',
        activation_uses_test_or_user_metrics=False,
        scope='Activation of this joint checkpoint; earlier cross-generator findings motivated the joint experiment'))
    print('Default photo detector now uses the verified joint generation model')


if __name__ == '__main__':
    main()
