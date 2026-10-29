import test from 'node:test';
import assert from 'node:assert/strict';
import { issuesFor, issueRanges } from '../web/review.mjs';

test('constant parallel displacement is flagged on every frame', () => {
  const m = {ops: Array(4).fill({op: 'copy'}), drift: Array(4).fill({sure: true, dt: 0, dx: 12, dy: 0, shift: 12}), exactness: {offset_px: [12, 0]}};
  const issues = issuesFor(m);
  assert.ok(issues.every(f => f.position));
  assert.deepEqual(issueRanges(issues, 'position'), [{start: 0, end: 3, key: 'position'}]);
});
test('unknown measurements are uncertain, not passed', () => {
  const issues = issuesFor({ops: [{op: 'copy'}, {op: 'interp'}, {op: 'copy'}], drift: [null, null, {sure: false, dt: 0, shift: 0}]});
  assert.equal(issues[0].uncertain, true);
  assert.equal(issues[1].gap, true);
  assert.equal(issues[2].uncertain, true);
  assert.equal(issueRanges(issues, 'uncertain').length, 2);
});
test('simultaneous timing and position problems remain visible', () => {
  const issues = issuesFor({ops: [{op: 'copy'}], drift: [{sure: true, dt: 1, shift: 8}]});
  assert.equal(issues[0].timing, true);
  assert.equal(issues[0].position, true);
});

test('unmeasurable frames do not flood candidate issue navigation', () => {
  const issues=issuesFor({ops:[{op:'copy'},{op:'copy'}],drift:[{sure:false,shift:0},{sure:false,shift:8}]});
  assert.deepEqual(issueRanges(issues),[{start:1,end:1,key:'position'}]);
  assert.equal(issueRanges(issues,'uncertain').length,1);
});

test('summary categories are disjoint and account for every frame', async () => {
  const {reviewCounts}=await import('../web/review.mjs');
  assert.deepEqual(reviewCounts([
    {gap:false,timing:false,position:true,uncertain:true},
    {gap:false,timing:false,position:false,uncertain:true},
    {gap:false,timing:false,position:false,uncertain:false}
  ]),{candidates:1,notMeasurable:1,consistent:1});
});

test('framing review separates measured residuals from whole-image phase offsets', async () => {
  const { framingSummary } = await import('../web/review.mjs');
  const framing={applied:true, rendered_series:[{frame:0,dx:1,dy:2}],rendered_motion:{headline:'No clear progressive drift',action:'Inspect residuals'}};
  const issues=issuesFor({ops:[{op:'copy'},{op:'copy'}],drift:[{sure:true,shift:40},{sure:true,shift:40}],framing});
  assert.equal(issues[0].position,false);
  assert.equal(issues[1].position,false);
  assert.equal(issues[1].uncertain,true);
  assert.match(framingSummary(framing),/^No clear progressive drift/);
  assert.match(framingSummary({motion:{state:'unknown',headline:'Not measurable'}}),/Not measurable/);
});

test('verdict gives a repair task without declaring an uncertain shot clean', async () => {
  const {shotVerdict}=await import('../web/review.mjs');
  const m={analysis_only:true,framing:{before_px:42,after_px:6,motion:{state:'stable'}},summary:{interpolated:[]}};
  assert.equal(shotVerdict(m).title,'Fixed framing mismatch');
  assert.equal(shotVerdict(m).task,'Correct framing');
  assert.equal(shotVerdict({...m,summary:{interpolated:[20]}}).task,'Repair timing + framing');
  assert.equal(shotVerdict({...m,summary:{interpolated:[20],segments:[{confidence:1}]}}).task,null);
  assert.equal(shotVerdict({...m,framing:{motion:{state:'unknown'}}}).title,'Insufficient evidence');
  assert.equal(shotVerdict({...m,framing:{...m.framing,motion:{state:'drift'}}}).task,null);
  assert.equal(shotVerdict({...m,framing:{...m.framing,applied:true}}).title,'Fixed framing corrected');
});

test('timing-only render reports completion without claiming spatial evidence', async () => {
  const {shotVerdict}=await import('../web/review.mjs');
  const m={repair_task:'Repair timing',summary:{complete:true,copied:58,interpolated:[20,40],segments:[{confidence:7}]}};
  const v=shotVerdict(m);
  assert.equal(v.title,'Timing repair rendered');
  assert.equal(v.state,'review');
  assert.match(v.evidence,/2 gap fills rendered; 58 source frames copied/);
  assert.match(v.evidence,/Framing was not measured/);
  assert.equal(v.task,null);
  for (const unsafe of [
    {...m,analysis_only:true},
    {...m,framing:{motion:{state:'unknown'}}},
    {...m,summary:{...m.summary,complete:false}},
    {...m,summary:{...m.summary,missing_plate_frames:[59]}},
    {...m,summary:{...m.summary,segments:[{confidence:1}]}}
  ]) assert.equal(shotVerdict(unsafe).title,'Insufficient evidence');
});

test('framing-only repair never claims that an uncertain timeline was repaired', async () => {
  const {shotVerdict}=await import('../web/review.mjs');
  const result=shotVerdict({timing_preserved:true,preserved_frames:193,
    framing:{applied:true,before_px:6.2,after_px:1.4,motion:{state:'stable'}},
    timing_analysis:{missing_plate_frames:[191],segments:[{confidence:-2}]}});
  assert.equal(result.title,'Fixed framing corrected; timing unchanged');
  assert.equal(result.state,'review');
  assert.match(result.evidence,/All 193 AI frames preserved/);
  assert.match(result.next,/Timing was not repaired or verified/);
  assert.equal(result.task,null);
});
