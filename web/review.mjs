// Pure review logic shared by the viewer and regression tests. Frames are zero-based.
export function issuesFor(manifest, threshold = 4) {
  const framing = manifest.framing;
  const samples = new Map((framing?.applied ? framing.rendered_series : framing?.series)?.map(s => [s.frame, s]) || []);
  return manifest.ops.map((op, i) => {
    const d = manifest.drift[i];
    const sample = samples.get(i);
    return {
      gap: op.op !== 'copy',
      timing: Boolean(d?.sure && d.dt),
      position: framing ? Boolean(sample && Math.hypot(sample.dx, sample.dy) > threshold) : d?.shift != null && d.shift > threshold,
      uncertain: op.op === 'copy' && (!d || !d.sure || (framing ? !sample : d.shift == null)),
    };
  });
}

export function issueRanges(issues, filter = 'all') {
  const ranges = [];
  issues.forEach((flags, i) => {
    const kinds = Object.keys(flags).filter(k => flags[k] && ((filter === 'all' && k !== 'uncertain') || filter === k));
    if (!kinds.length) return;
    const key = kinds.join(', '), last = ranges.at(-1);
    if (last && last.end === i - 1 && last.key === key) last.end = i;
    else ranges.push({ start: i, end: i, key });
  });
  return ranges;
}

export function reviewCounts(issues) {
  return { candidates: issues.filter(f => f.gap || f.timing || f.position).length,
    notMeasurable: issues.filter(f => f.uncertain && !(f.gap || f.timing || f.position)).length,
    consistent: issues.filter(f => !Object.values(f).some(Boolean)).length };
}
export function timingExplanation(d) {
  if (!d) return 'No matched frame to measure';
  if (!d.sure) return d.timing_reason || 'Timing not distinguishable from neighbouring frames';
  return d.dt ? `May be ${Math.abs(d.dt)} frame${Math.abs(d.dt)===1?'':'s'} ${d.dt>0?'ahead':'behind'}` : 'Timing consistent with plate';
}

export function framingSummary(f) {
  if (!f) return null;
  const motion = f.rendered_motion || f.motion;
  const lines = [motion?.headline || "Drift has not been established"];
  if (f.scale_ai_to_plate != null) {
    lines.push(f.applied ? "FIXED FRAMING CORRECTED" : "FIXED FRAMING · correction preview only");
    lines.push(`One constant scale: ${((f.scale_ai_to_plate - 1) * 100).toFixed(2)}% · plus position`);
    lines.push(`Shared-feature error: ${f.before_px} → ${f.after_px} px on withheld frames`);
    if (f.rendered_error_px != null) lines.push(`Decoded render: ${f.rendered_error_px} px typical remaining error`);
    if (motion?.start_to_end_px != null) lines.push(`Start → end movement: ${motion.start_to_end_px} px\n95% residual variation: ${motion.variation_p95_px} px\nMatching noise estimate: ${motion.noise_floor_px} px`);
    lines.push("Measurements in plate pixels, on shared detail. Local deformation is not ruled out.");
  }
  lines.push(f.applied ? motion?.action : f.scale_ai_to_plate ? "Next: Repair → Correct framing (scale + position), then inspect the comparison." : motion?.action);
  return lines.filter(Boolean).join("\n\n");
}

// A verdict is evidence about this shot, not a guarantee that every pixel matches.
export function shotVerdict(m) {
  const f=m.framing, motion=f?.rendered_motion || f?.motion;
  const s=m.summary || {};
  const gaps=s.interpolated?.length || 0, missing=s.missing_plate_frames?.length || 0;
  const timing=Boolean(gaps || s.dropped_gen_frames?.length || s.offset_median);
  const weak=timing && s.segments?.some(x => x.confidence == null || x.confidence < 2);
  if (m.timing_preserved && f?.applied) {
    return {title:'Fixed framing corrected; timing unchanged',state:'review',
      evidence:`${f.before_px} → ${f.after_px} px shared-feature error on withheld frames. All ${m.preserved_frames} AI frames preserved.`,
      next:'Timing was not repaired or verified. Inspect the original sequence and any remaining motion or local differences.',task:null};
  }
  // A timing-only render has no framing fit. Describe the completed operation
  // without treating an absent spatial check as either failure or a clean shot.
  if (!m.analysis_only && m.repair_task === 'Repair timing' && !f && s.complete && !missing && !weak) {
    return {title:'Timing repair rendered', state:'review',
      evidence:`${gaps} gap fills rendered; ${s.copied ?? 0} source frames copied. Framing was not measured in this repair.`,
      next:'Inspect the filled frames and source correspondence. Run Analyse to assess framing.', task:null};
  }
  const evidence=f?.before_px != null
    ? `${f.before_px} → ${f.after_px} px shared-feature error on withheld frames${f.applied ? '. Correction applied.' : '. Predicted correction; originals unchanged.'}`
    : 'There is not enough reliable spatial evidence for a framing verdict.';
  let title='Insufficient evidence', state='unknown', next='Choose a mask over shared subject detail and analyse again.', task=null;
  if (motion?.state === 'drift') {
    title='Possible progressive drift';state='review';next='Inspect the residual trend and the largest offsets. One fixed correction cannot remove progressive movement.';
  } else if (motion?.state === 'wobble') {
    title='Possible wobble';state='review';next='Inspect the peaks on the movement graph. Local deformation or tracking noise may contribute.';
  } else if (motion?.state === 'stable') {
    const mismatch=f.before_px>4 && f.after_px<f.before_px*.85;
    title=f.applied ? 'Fixed framing corrected' : mismatch ? 'Fixed framing mismatch' : 'No clear framing mismatch';
    state='measured';
    task=!f.applied && mismatch ? (timing ? 'Repair timing + framing' : 'Correct framing') : timing && m.analysis_only ? 'Repair timing' : null;
    next=f.applied ? 'Compare Plate and Output in Overlay or Flicker. No clear progressive drift was measured; local differences remain possible.'
      : task ? `Next: ${task}. Inspect the result before using it.` : 'No framing repair is indicated by the sampled measurements. Inspect local differences visually.';
  }
  if (missing) {
    next=`${missing} frames have no bounded match. Inspect missing footage first. LTX reconstruction is experimental.`;task=null;
  } else if (weak) {
    next='Timing edits have weak evidence. Inspect neighbouring frames before allowing a candidate repair.';task=null;
  } else if (!task && timing && m.analysis_only && motion?.state !== 'stable') {
    next+=` The timing plan also proposes ${gaps} gap fills; review it separately.`;
  }
  return {title,state,evidence,next,task};
}
