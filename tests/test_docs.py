from pathlib import Path


def test_readme_uses_test_acc_flag():
    readme = Path('Face Emotion Recognition/README.md').read_text(encoding='utf-8')
    assert '--test_acc' in readme
    assert '--test_cc' not in readme
