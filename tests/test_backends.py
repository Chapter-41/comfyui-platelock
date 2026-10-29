import unittest
from unittest.mock import patch
from platelock.backends import LocalLTX, load_rife, preflight_ltx
import types


class BackendTests(unittest.TestCase):
    def test_missing_models_fail_before_generation(self):
        folders=types.SimpleNamespace(get_full_path=lambda *a:None)
        with patch.dict('sys.modules',{'folder_paths':folders}):
            with self.assertRaisesRegex(ValueError,'local models'):
                preflight_ltx()

    def test_integrated_ltx_passes_window_dimensions_and_context(self):
        calls=[]
        with patch('platelock.backends.preflight_ltx'):
            backend=LocalLTX('same shot',seed=42)
        def call(name,**kw):
            calls.append((name,kw))
            return tuple(f'{name}:{i}' for i in range(2 if name == 'LTXVConditioning' else 3))
        backend.call=call
        output=backend('context',320,192,33,25.)
        byname=dict(calls)
        self.assertEqual(byname['EmptyLTXVLatentVideo'],dict(width=320,height=192,length=33,batch_size=1))
        self.assertEqual(byname['LTXVPreprocess']['image'],'context')
        self.assertEqual(byname['LTXVEmptyLatentAudio']['frames_number'],33)
        self.assertEqual(byname['RandomNoise']['noise_seed'],42)
        self.assertEqual(byname['VAEDecodeTiled']['samples'],'LTXVCropGuides:2')
        self.assertEqual(output,'VAEDecodeTiled:0')
