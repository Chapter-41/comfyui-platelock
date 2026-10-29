import unittest
import cv2
import numpy as np
from platelock.framing import analyse_framing, motion_verdict, correction_matrix
from platelock.plan import Op

class FramingTest(unittest.TestCase):
    def test_motion(self):
        frames=np.arange(0,100,2)
        self.assertEqual(motion_verdict(frames,np.tile([30,-20],(50,1)),100)['state'],'stable')
        self.assertEqual(motion_verdict(frames,np.column_stack([frames*.3,frames*0]),100)['state'],'drift')
        self.assertEqual(motion_verdict(frames,np.column_stack([15*np.sin(frames/8),frames*0]),100)['state'],'wobble')
        self.assertEqual(motion_verdict([],[],100)['state'],'unknown')
        self.assertEqual(motion_verdict(frames[:10],np.zeros((10,2)),100)['state'],'unknown')

    def test_scale_translation_and_validation(self):
        rng=np.random.default_rng(4)
        image=rng.integers(0,256,(360,640,3),dtype=np.uint8)
        image=cv2.GaussianBlur(image,(3,3),0)
        matrix=np.float32([[.98,0,-3],[0,.98,5]])
        shifted=cv2.warpAffine(image,matrix,(640,360))
        fit=analyse_framing([image]*24,[shifted]*24,[Op(i,'copy',i) for i in range(24)],640)
        self.assertAlmostEqual(fit['scale_ai_to_plate'],1/.98,places=3)
        self.assertLess(fit['after_px'],.5)
        self.assertGreater(fit['before_px'],5)
        recovered=correction_matrix(fit,1280,720,640/360)
        self.assertAlmostEqual(recovered[0,2],6/.98,delta=.5)
        self.assertAlmostEqual(recovered[1,2],-10/.98,delta=.5)
        self.assertEqual(fit['motion']['state'],'stable')

    def test_no_features_rejected(self):
        image=np.zeros((144,256,3),np.uint8)
        with self.assertRaisesRegex(ValueError,'shared detail'):
            analyse_framing([image]*20,[image]*20,[Op(i,'copy',i) for i in range(20)],256)

    def test_moving_scene_with_camera_motion_and_changed_colour(self):
        from platelock.demo import examples
        plate,cases,_=examples()
        fit=analyse_framing(plate,cases['framing'],[Op(i,'copy',i) for i in range(len(plate))],320,max_samples=30)
        self.assertGreater(fit['validation_frames'],10)
        self.assertAlmostEqual(fit['scale_ai_to_plate'],1/.975,delta=.003)
        self.assertLess(fit['after_px'],fit['before_px']*.25)
        self.assertEqual(fit['motion']['state'],'stable')

    def test_progressive_motion_is_not_disguised_by_fixed_correction(self):
        from platelock.demo import moving_plate
        plate=moving_plate()
        ai=np.stack([cv2.warpAffine(f,np.float32([[1,0,5+i*.4],[0,1,0]]),(320,180)) for i,f in enumerate(plate)])
        fit=analyse_framing(plate,ai,[Op(i,'copy',i) for i in range(len(plate))],320,max_samples=30)
        self.assertEqual(fit['motion']['state'],'drift')
        self.assertGreater(fit['motion']['start_to_end_px'],12)

    def test_unrelated_moving_scene_is_not_confidently_corrected(self):
        from platelock.demo import moving_plate
        with self.assertRaisesRegex(ValueError,'shared detail'):
            analyse_framing(moving_plate(seed=41),moving_plate(seed=91),[Op(i,'copy',i) for i in range(60)],320,max_samples=24)

    def test_fixed_framing_does_not_invent_timing_edits(self):
        from platelock.demo import examples
        from platelock.core import analyze_arrays
        plate,cases,_=examples()
        gray=lambda frames: np.stack([cv2.cvtColor(f,cv2.COLOR_RGB2GRAY) for f in frames]).astype(np.float32)/255
        result=analyze_arrays(gray(cases['framing']),gray(plate))
        self.assertEqual(result.summary['interpolated'],[])
        self.assertEqual(result.summary['dropped_gen_frames'],[])
        self.assertEqual(result.summary['missing_plate_frames'],[])
        self.assertTrue(all(o.plate==o.gen for o in result.ops))
