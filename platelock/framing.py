"""One fixed scale/translation for a shot, validated on withheld frames.

No per-frame stabilisation: residual changes remain visible. All reported
measurements are in original plate pixels, on shared tracked detail only.
"""
import cv2
import numpy as np


def correspondences(plate, ai, mask=None, width=640):
    h = round(plate.shape[0]*width/plate.shape[1])
    gray = lambda f: cv2.cvtColor(cv2.resize(f,(width,h)),cv2.COLOR_RGB2GRAY)
    a,b=gray(plate),gray(ai)
    roi=None if mask is None else (cv2.resize(mask.astype(np.float32),(width,h))>.2).astype(np.uint8)*255
    sift=cv2.SIFT_create(nfeatures=1200,contrastThreshold=.025)
    ka,da=sift.detectAndCompute(a,roi); kb,db=sift.detectAndCompute(b,None)
    if da is None or db is None or min(len(da),len(db))<10: return None
    matcher=cv2.BFMatcher()
    forward=matcher.knnMatch(da,db,k=2); reverse=matcher.match(db,da)
    good=[pair[0] for pair in forward if len(pair)==2 and pair[0].distance<.72*pair[1].distance
          and reverse[pair[0].trainIdx].trainIdx==pair[0].queryIdx]
    if len(good)<10: return None
    p=np.float32([ka[m.queryIdx].pt for m in good]);q=np.float32([kb[m.trainIdx].pt for m in good])
    m,ok=cv2.estimateAffinePartial2D(p,q,method=cv2.RANSAC,ransacReprojThreshold=1.5,maxIters=3000,confidence=.995)
    if m is None or ok.sum()<10 or ok.mean()<.5: return None
    p,q=p[ok.ravel().astype(bool)],q[ok.ravel().astype(bool)]
    if np.ptp(p,axis=0).min()<min(h,width)*.12: return None
    # Fit scale and translation only; do not quietly add rotation or shear.
    pc,qc=p.mean(axis=0),q.mean(axis=0)
    z=float(((p-pc)*(q-qc)).sum()/((p-pc)**2).sum());shift=qc-z*pc
    err=np.linalg.norm(p*z+shift-q,axis=1)
    if not .9<z<1.1 or np.median(err)>1: return None
    return p/width,q/width,z,shift/width,float(np.median(err)/width)


def motion_verdict(frames, offsets, total, noise_px=2):
    frames=np.asarray(frames);xy=np.asarray(offsets,dtype=float)
    if len(xy)<8 or frames.max()-frames.min()<total*.65:
        return {'state':'unknown','headline':'Not enough shared detail to judge drift',
                'action':'Choose a mask over the subject and analyse again.'}
    center=np.median(xy,axis=0);deviation=np.linalg.norm(xy-center,axis=1)
    bins=[]
    for lo,hi in zip(np.linspace(0,total,6)[:-1],np.linspace(0,total,6)[1:]):
        values=xy[(frames>=lo)&(frames<hi)]
        bins.append(np.median(values,axis=0) if len(values)>=2 else None)
    usable=[(i,v) for i,v in enumerate(bins) if v is not None]
    if len(usable)<4:
        return {'state':'unknown','headline':'Coverage is too uneven to judge drift',
                'action':'Inspect the unmeasured part of the shot.'}
    x=np.array([i for i,v in usable]);y=np.array([v for i,v in usable])
    fitted=np.column_stack([np.polyval(np.polyfit(x,y[:,j],1),x) for j in range(2)])
    r2=1-float(((y-fitted)**2).sum())/(float(((y-y.mean(axis=0))**2).sum())+1e-8)
    end=float(np.linalg.norm(y[-1]-y[0]));spread=float(np.percentile(deviation,95))
    threshold=max(4.,3*noise_px)
    state='drift' if end>threshold and r2>.7 else 'wobble' if spread>threshold else 'stable'
    text={'drift':('Progressive movement remains','Inspect the trend before applying any per-frame stabilisation.'),
          'wobble':('Possible wobble remains; no clear progressive drift','Inspect the largest residuals. Matching noise and local deformation may contribute.'),
          'stable':('Mostly fixed framing; no clear progressive drift','Inspect the corrected comparison. No per-frame stabilisation is indicated by these measurements.')}
    return {'state':state,'headline':text[state][0],'action':text[state][1],
            'residual_offset_px':np.round(center,2).tolist(),'variation_p95_px':round(spread,2),
            'start_to_end_px':round(end,2),'noise_floor_px':round(noise_px,2),
            'trend_r2':round(r2,3),'windows':[None if v is None else np.round(v,2).tolist() for v in bins]}


def analyse_framing(plate, ai, ops, plate_width, mask=None, max_samples=80):
    candidates=[o for o in ops if o.op=='copy' and o.gen<len(ai) and o.plate<len(plate)]
    selected=[candidates[i] for i in np.unique(np.linspace(0,len(candidates)-1,min(max_samples,len(candidates))).astype(int))] if candidates else []
    matches=[]
    for o in selected:
        match=correspondences(plate[o.plate],ai[o.gen],mask)
        if match is not None: matches.append((o.plate,*match))
    if len(matches)<12 or matches[-1][0]-matches[0][0]<len(ops)*.65:
        raise ValueError('Not enough shared detail across the shot for a fixed framing correction. Select a subject mask.')
    train=matches[::2];test=matches[1::2]
    z=float(np.median([m[3] for m in train]));t=np.median([m[4] for m in train],axis=0)
    before=[];after=[];series=[]
    for f,p,q,_,_,err in matches:
        corrected=(q-t)/z
        offset=np.median(corrected-p,axis=0)*plate_width
        series.append({'frame':f,'dx':round(float(offset[0]),2),'dy':round(float(offset[1]),2),
                       'error_px':round(err*plate_width,2),'matches':len(p)})
    for f,p,q,*_ in test:
        before.append(float(np.median(np.linalg.norm(q-p,axis=1)))*plate_width)
        after.append(float(np.median(np.linalg.norm((q-t)/z-p,axis=1)))*plate_width)
    b,a=float(np.median(before)),float(np.median(after))
    if b>4 and a>b*.85:
        raise ValueError('The fixed scale/position correction did not improve withheld frames enough; no correction rendered.')
    noise=float(np.median([m[5]*plate_width for m in matches]))
    verdict=motion_verdict([s['frame'] for s in series],[[s['dx'],s['dy']] for s in series],len(ops),noise)
    return {'scale_ai_to_plate':1/z,'translation_normalized':(-t/z).tolist(),
            'ai_scale':z,'fit_frames':len(train),'validation_frames':len(test),'sampled_frames':len(selected),
            'before_px':round(b,2),'after_px':round(a,2),'series':series,'motion':verdict,
            'scope':'One fixed scale and translation; no per-frame stabilisation.',
            'measurement':'Median shared-feature displacement on withheld frames; not whole-image pixel equality.'}


def correction_matrix(fit,width,height,plate_aspect):
    # Fit coordinates are normalised by width in both axes.
    return np.float32([[fit['scale_ai_to_plate'],0,fit['translation_normalized'][0]*width],
                       [0,fit['scale_ai_to_plate'],fit['translation_normalized'][1]*height*plate_aspect]])
