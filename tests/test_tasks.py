import unittest
from platelock.tasks import resolve_task, require_framing_only_plan

class TasksTest(unittest.TestCase):
    def test_presets_and_legacy(self):
        self.assertEqual(resolve_task('Correct framing','LTX','Measure only'),('None','Correct framing (scale + position)'))
        self.assertEqual(resolve_task('Repair timing','None','Correct constant shift'),('RIFE','Measure only'))
        self.assertEqual(resolve_task('Repair timing + framing','None','Measure only'),('RIFE','Correct framing (scale + position)'))
        self.assertEqual(resolve_task('Advanced settings','None','Correct constant shift'),('None','Correct constant shift'))
        self.assertEqual(resolve_task('Reconstruct missing footage (experimental)','None','Measure only')[0],'LTX')
        with self.assertRaises(ValueError): resolve_task('surprise','RIFE','Measure only')

    def test_framing_does_not_silently_retime(self):
        clean={'interpolated':[],'missing_plate_frames':[],'dropped_gen_frames':[],'offset_median':0}
        require_framing_only_plan('Correct framing',clean)
        for key,value in [('interpolated',[3]),('missing_plate_frames',[0]),('dropped_gen_frames',[4]),('offset_median',2)]:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError,'also needs timing'):
                require_framing_only_plan('Correct framing',{**clean,key:value})
        require_framing_only_plan('Repair timing + framing',{**clean,'interpolated':[3]})
