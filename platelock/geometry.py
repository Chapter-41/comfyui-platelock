"""Conservative similarity fit and appearance residuals in plate coordinates."""
import cv2
import numpy as np


def inspect_geometry(reference, candidate, roi, scale):
    # Fit edges, not RGB: appearance is expected to change in an AI restyle.
    a = cv2.normalize(reference, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    b = cv2.normalize(candidate, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    points = cv2.goodFeaturesToTrack(a, 120, .02, 8, mask=(roi > .2).astype(np.uint8))
    if points is None or len(points) < 12:
        return None
    q, ok, _ = cv2.calcOpticalFlowPyrLK(a, b, points, None, winSize=(21,21), maxLevel=3)
    back, ok2, _ = cv2.calcOpticalFlowPyrLK(b, a, q, None, winSize=(21,21), maxLevel=3)
    keep = ok.ravel().astype(bool) & ok2.ravel().astype(bool) & (np.linalg.norm(back-points,axis=2).ravel()<.7)
    if keep.sum() < 12:
        return None
    m, inliers = cv2.estimateAffinePartial2D(points[keep],q[keep],method=cv2.RANSAC,ransacReprojThreshold=1)
    if m is None or inliers.mean()<.7:
        return None
    zoom = float(np.hypot(m[0,0],m[1,0]))
    angle = float(np.degrees(np.arctan2(m[1,0],m[0,0])))
    if not .9 < zoom < 1.1 or abs(angle)>5:
        return None
    h,w = a.shape
    aligned = cv2.warpAffine(candidate,m,(w,h),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP)
    # Residuals are appearance differences, never automatically defects.
    regions=[]
    diff=np.abs(reference-aligned)
    for y in range(3):
        for x in range(4):
            tile=diff[y*h//3:(y+1)*h//3,x*w//4:(x+1)*w//4]
            region=roi[y*h//3:(y+1)*h//3,x*w//4:(x+1)*w//4]
            score=float((tile*region).sum()/(region.sum()+1e-8))
            if region.mean()>.2 and score>.65:
                regions.append({"x":x/4,"y":y/3,"w":.25,"h":1/3,"score":round(score,2)})
    return {"rotation_deg":round(angle,3),"scale":round(zoom,5),
            "inliers":int(inliers.sum()),"regions":regions,
            "note":"Appearance differences after alignment; intentional changes may be highlighted."}
