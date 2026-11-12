import type {ResearchEvidenceData} from './ResearchEvidence';
import type {QualityProfile} from './QualityAnalytics';
import type {WeightedScore} from './ExperimentalScore';
export type RecordData<T> = {id:string;kind:string;created_at:string;payload:T};
export type Box = {class:string;bbox:number[];confidence?:number;outcome?:string};
export type Dataset = {name:string;sha256:string;version:string;modality:string;profile:{image_count?:number;row_count?:number;object_count?:number;class_counts?:Record<string,number>;annotation_format?:string;issues?:unknown[];[key:string]:unknown};training:{eligible:boolean;reasons:string[]};provenance:Record<string,unknown>;archived?:boolean};
export type Split = {dataset_id:string;checksum:string;images:Record<string,string[]>;independence:string;near_duplicate_pairs:number;group_column:string|null};
export type Context = {original_text:string;goal:{task_type:string;[key:string]:unknown};clarification_questions:string[];confirmed:boolean;provider:string;provider_error?:string|null;provider_error_code?:string|null;[key:string]:unknown};
export type Metric = {metric_id:string;dimension:string;state:string;value:unknown;reason:string;evidence:unknown;assessment_status?:string};
export type Certificate = {overall_status:string;executive_summary:string;metrics:Metric[];findings:{requirement:{description:string;expected_value:unknown;metric_id:string};state:string;reason:string}[];recommendations:{description:string}[];limitations:string[];reproducibility:unknown;quality_profile?:QualityProfile|null;quality_measurements?:{research_evidence?:ResearchEvidenceData;distributions:Record<string,{count:number;mean:number;bins:{lower:number;upper:number;count:number}[]}>};experimental_score?:WeightedScore|null;[key:string]:unknown};
export type Job = {task:string;status:string;stage:string;config:{dataset_id?:string;split_id?:string;[key:string]:unknown};error:string|null;result:Certificate|{model_id?:string;evaluation_ids?:string[]}|null;progress?:{epoch?:number;epochs?:number;[key:string]:unknown}};
export type Compute = {torch_installed:boolean;detector_installed:boolean;cuda_available:boolean;devices:string[];selected_device:string;gpu_name?:string;torch_version?:string;error?:string};
export type Model = {job_id:string;dataset_id:string;readiness_id:string;configuration:{architecture:string;seed:number;model_type:string;classes:string[];[key:string]:unknown};validation:Record<string,number>;artifact:string;artifact_sha256:string;training_images:number;[key:string]:unknown};
export type Evaluation = {per_class?:Record<string,Record<string,number|null>>;curves?:{x_label:string;y_label:string;x:number[];series:{class:string;values:number[]}[]}[];confusion_matrix?:{labels:string[];counts:number[][];method:string};error_denominators?:Record<string,string>;environment?:unknown;model_id:string;dataset_id:string;test_sha256:string;metrics:Record<string,number|null|Record<string,number>>;settings:Record<string,unknown>;[key:string]:unknown};
export type Sample = {image:string;url:string;annotations:Box[];[key:string]:unknown};
export type Prediction = {image:string;ground_truth:Box[];predictions:Box[];missed:Box[]};
export type ResearchRow = {dataset:string;dataset_id:string;dataset_version:string;model_id:string;configuration:{seed:number;architecture:string;[key:string]:unknown};readiness_status:string;experimental_quality_score?:number|null;experimental_score_coverage?:number|null;training_images:number;test_dataset:string;test_sha256:string;configuration_group:string;seed:number;metrics:Record<string,number|null>;generalization_gap:number|null;artificial:boolean;quality_metrics:Record<string,number>;date:string};
export type Research = {predictive_validity?:unknown[];intervention_pairs?:unknown[];rows:ResearchRow[];metric_options:string[];selected_metric:string|null;correlations:{configuration_group:string;independent_datasets:number;pearson:number|null;spearman:number|null;pearson_confidence_interval_95?:number[]|null;spearman_confidence_interval_95?:number[]|null;interval_method?:string;interpretation:string;points:{dataset:string;x:number;y:number;seeds:number;interval:[number,number]|null}[]}[];summary:string[];aggregates:unknown[];performance_matrix:{model_id:string;test_dataset:string;mAP50_95:number}[]};
export async function api<T>(path:string, options:RequestInit={}):Promise<T>{
  const response=await fetch('/api/v1'+path,options);
  if(!response.ok){const body=await response.json().catch(()=>({detail:response.statusText}));throw new Error(typeof body.detail==='string'?body.detail:JSON.stringify(body.detail));}
  return response.json() as Promise<T>;
}
export const json=(body:unknown,method='POST'):RequestInit=>({method,headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
export const fmt=(value:unknown):string=>value===null||value===undefined?'Unavailable':typeof value==='number'?(Number.isInteger(value)?value.toLocaleString():value.toFixed(4)):typeof value==='object'?JSON.stringify(value):String(value);
export const completed=(job:RecordData<Job>)=>job.payload.task==='readiness'&&job.payload.status==='completed';
