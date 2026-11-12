"""Stable property IDs and precise scope. Registration is owned by metrics.py."""

# ID, name, dimension, required inputs, method, limitations
PROPERTIES = [
('UAV-01','Lighting diversity','coverage','brightness, contrast; optional lighting metadata','Luminance/contrast quantiles and declared category frequencies','Brightness does not establish day/night or weather.'),
('UAV-02','Weather diversity','coverage','weather metadata or configured classifier','Observed category counts and missingness; optional classifier estimates','Declared metadata is not independently authenticated; visual predictions remain estimates.'),
('UAV-03','Time-of-day coverage','coverage','timestamp or recorded time_of_day metadata','UTC timestamp hour counts or declared time categories','UTC hour is not solar/local time; appearance cannot establish capture time.'),
('UAV-04','Seasonal variation','coverage','season metadata or timestamp plus hemisphere','Declared season counts or explicitly configured meteorological month mapping','Hemisphere/calendar assumptions may not describe local climate.'),
('UAV-05','Scene/background diversity','coverage','scene metadata or local scene classifier/encoder','Category counts; bounded normalized embedding centroid dispersion','Image embeddings include objects and may encode irrelevant variation.'),
('UAV-06','Camera viewpoint diversity','coverage','viewpoint or calibrated camera orientation metadata','Category counts or angle quantiles with declared units','No precise camera orientation inferred from uncalibrated appearance.'),
('UAV-07','Flight altitude coverage','coverage','altitude metadata and units','Finite altitude quantiles, missing/invalid counts, declared intervals','Altitude datum and units must be declared; object size is not altitude.'),
('UAV-08','Object-scale diversity','coverage','valid normalized boxes and decoded dimensions','Per-class box width/height/area/aspect/occupancy quantiles and user bins','Annotation-derived only; unannotated objects are excluded.'),
('UAV-09','Object appearance diversity','coverage','annotated object crops and configured local encoder','Within-class mean cosine distance to normalized centroid','Encoder-dependent estimates; diversity does not prove usefulness.'),
('UAV-10','Occlusion coverage','coverage','object occlusion annotations','Per-class observed occlusion code counts and missingness','Missing annotation is unknown, not zero occlusion.'),
('UAV-11','Truncation coverage','coverage','truncation annotations or valid boxes','Per-class truncation counts and border-contact fractions','Border contact is only a truncation proxy.'),
('UAV-12','Background clutter','coverage','luminance entropy and valid annotations','Thumbnail entropy and annotated object density summaries','Entropy and object density are supporting proxies, not semantic background clutter.'),
('UAV-13','Object density and crowding','coverage','boxes, decoded dimensions, annotation presence','Objects/image and megapixel; bounded pair IoU and center-distance quantiles','Pair and image populations are explicit; no universal sparse/dense cutoffs.'),
('UAV-14','Image clarity and visual information','integrity','decoded pixels and encoded file sizes','Resolution, luminance, contrast, thumbnail Laplacian variance and bytes/pixel','No-reference proxies depend on content, sensor and thumbnail scale; bytes/pixel is not compression quality.'),
('UAV-15','Annotation correctness and completeness','label_quality','annotation parser evidence; optional independent audit','Reuse syntax/presence/manifest diagnostics and independently reviewed audit counts','Syntax is not semantic correctness or exhaustive visible-object annotation.'),
('UAV-16','Discriminative and challenging examples','label_quality','label-aware crop embeddings','Sampled nearest cross-class cosine similarity and candidate review pairs','Similarity is model-dependent; difficult examples are not automatically wrong or useful.'),
('UAV-17','Deployment representativeness','representativeness','explicit corresponding reference columns or category distributions','Reuse Wasserstein/JS comparisons; JS divergence for declared categorical references','Descriptive shift does not predict accuracy loss or prove deployment representativeness.'),
('UAV-18','Sample sufficiency and independence','duplication','class, hash, split and flight/session identifiers','Reuse class/duplicate/split/group statistics; count known collection groups','Group IDs are declared; total frames are not independent observations; adequacy is unverified.'),
('UAV-19','Sensor/modality integrity','integrity','sensor_types, units, calibration/alignment metadata','Validate declared mapped fields and units; paired sensor timestamp deltas','Calibration presence is not calibration validity; synchronization needs timestamps and explicit tolerance.'),
('UAV-20','Motion and temporal integrity','integrity','timestamp and sequence/frame/track identifiers','Within-sequence timestamp differences in recorded order, duplicates and frame gaps','Static imagery is not video; tracking identity correctness and physical motion require additional evidence.'),
]

PROPERTY_IDS = [item[0] for item in PROPERTIES]

ENHANCEMENTS = {
    'UAV-08':'native/model-input sides and area under explicitly configured letterbox/stretch geometry; separate object/image size-bin denominators',
    'UAV-12':'cached global edges and crowding; bounded object-local background-ring entropy/edges/texture; sample-relative candidate ranks',
    'UAV-14':'separate endpoint exposure, high-frequency residual and local contrast indicators; bounded object/ring clarity and contrast distributions',
    'UAV-15':'categorized one-to-one annotation comparison against explicitly independent complete references; negative/unknown/candidate evidence separated',
    'UAV-17':'shared three-valued image/object scenario conjunctions with explicit deployment-reference comparison and unknown coverage',
    'UAV-18':'cached dHash bounded BK-tree candidates and declared collection group/size/class/cross-split evidence; no effective sample-size estimator',
}

def register(metric, calculations, registry):
    # Lazy import prevents a cycle through the legacy metrics entry points.
    from metrics.uav_metrics.assessment import calculate
    image_only={'UAV-01','UAV-05','UAV-08','UAV-09','UAV-10','UAV-11','UAV-12','UAV-13','UAV-14','UAV-15','UAV-16'}
    detection_only={'UAV-08','UAV-09','UAV-10','UAV-11','UAV-13','UAV-15','UAV-16'}
    for mid,name,dimension,inputs,method,limits in PROPERTIES:
        if mid in ENHANCEMENTS:
            method += '; '+ENHANCEMENTS[mid]
            limits += ' Added proxies are descriptive; sampled scope, unavailable evidence and method limitations are retained in structured submeasurements.'
        def implementation(ds,g,p,r,mid=mid):
            return calculate(mid,ds,g,p,r)
        implementation.__doc__=name+'. '+method
        metric(mid,dimension,method,modalities=('image',) if mid in image_only else ('image','tabular'),name=name,
            assessment_category='UAV dataset property',required_dataset_fields=inputs.split('; '),
            applicable_tasks=['object_detection','tracking'] if mid in detection_only else ['object_detection','tracking','telemetry','classification'],
            measurement_type='mixed observed/declared/proxy/optional estimated',output_type='object',
            applicability='Selected according to task, modality and available evidence; missing evidence is explicit',
            evidence_requirements=inputs,limitations=[limits],implementation_version='2.0' if mid in ('UAV-08','UAV-12','UAV-14','UAV-15','UAV-17','UAV-18') else '1.0',
            implementation_status='optional_local_model_adapter' if mid in ('UAV-09','UAV-16') else 'implemented')(implementation)
        registry[mid].unit='structured; component units reported in value'
