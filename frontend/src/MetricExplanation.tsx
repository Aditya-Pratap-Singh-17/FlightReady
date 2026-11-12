import React from 'react';
import type {QualityMetric} from './QualityAnalytics';
import {fmt} from './api';

export function MetricExplanation({metric:m}:{metric:QualityMetric}){
  return <div className="fr-metric-explanation"><dl><dt>Definition</dt><dd>{String(m.definition.label||m.label)}. Direction: {String(m.definition.direction||'See measurement definition')}.</dd><dt>Observed result</dt><dd>{fmt(m.raw_value)}. {m.score===null?'Not scored':m.score.toFixed(2)+'/100 criterion attainment'}.</dd><dt>Interpretation</dt><dd>{m.reason}</dd><dt>Evidence and population</dt><dd><pre>{JSON.stringify(m.evidence,null,2)||'Evidence was not recorded.'}</pre></dd><dt>Method and limitations</dt><dd>{m.normalization}. {String(m.definition.limitations||'Refer to the original metric registry for population, units and denominator.')}</dd><dt>Recommended action</dt><dd>{m.recommendation||'Review against the defined goal; this measurement alone does not predict model success.'}</dd></dl><details><summary>Exact weights, normalization and provenance</summary><pre>{JSON.stringify(m,null,2)}</pre></details></div>;
}
