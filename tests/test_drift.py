import unittest
import numpy as np
import cv2
from platelock.drift import measure
from test_platelock import synthetic_plate


def rgb(a):
    return np.repeat((a*255).clip(0,255).astype(np.uint8)[...,None],3,axis=3)


class EvidenceTests(unittest.TestCase):
    def test_identical_moving_plate_is_consistent(self):
        p=rgb(synthetic_plate(n=20))
        d=measure(p,p,[True]*len(p),p.shape[2])
        self.assertTrue(all(x['sure'] and x['dt']==0 for x in d))
        self.assertTrue(all(x['shift']<.1 for x in d))

    def test_static_is_not_an_error_or_verified_timing(self):
        p=rgb(synthetic_plate(n=12));p[:]=p[0]
        d=measure(p,p,[True]*len(p),p.shape[2])
        self.assertTrue(all(not x['sure'] and x['dt']==0 for x in d))
        self.assertTrue(all('motion' in x['timing_reason'] for x in d))

    def test_parallel_displacement_in_plate_pixels(self):
        p=rgb(synthetic_plate(n=20))
        shifted=np.stack([cv2.warpAffine(f,np.float32([[1,0,5],[0,1,-3]]),(256,144),borderMode=cv2.BORDER_REFLECT) for f in p])
        d=measure(p,shifted,[True]*len(p),512)
        valid=[x for x in d if x['shift'] is not None]
        self.assertGreater(len(valid),15)
        self.assertAlmostEqual(float(np.median([x['dx'] for x in valid])),10,delta=1)
        self.assertAlmostEqual(float(np.median([x['dy'] for x in valid])),-6,delta=1)

    def test_blank_and_invalid_frames_never_pass(self):
        p=np.zeros((6,64,96,3),np.uint8)
        d=measure(p,p,[True,True,False,True,True,True],96)
        self.assertIsNone(d[2])
        self.assertTrue(all(x is None or (not x['sure'] and x['shift'] is None) for x in d))

    def test_empty_mask_is_rejected(self):
        p=rgb(synthetic_plate(n=10))
        with self.assertRaisesRegex(ValueError,'mask'):
            measure(p,p,[True]*len(p),256,np.zeros(p.shape[1:3]))
