

// -----------------------------------------------------------------------------
// 1. SETTINGS AND DATA
// -----------------------------------------------------------------------------

var year = 2023;
var startDate = ee.Date.fromYMD(year, 1, 1);
var endDate = startDate.advance(1, "year");
var scale = 5;
var seed = 30;                 // Retained from the supplied original script.
var trainingFraction = 0.8;
var labelProperty = "Height";

var s2Col = ee.ImageCollection("COPERNICUS/S2_HARMONIZED");
var s2CloudScoreCol = ee.ImageCollection(
  "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"
);
var cloudScoreBand = "cs_cdf";
var clearThreshold = 0.60;

var planetCol = ee.ImageCollection(
  "projects/planet-nicfi/assets/basemaps/asia"
);
var s1Col = ee.ImageCollection("COPERNICUS/S1_GRD");

var roi = ee.FeatureCollection(
  "projects/your-project/assets/study_area"
);
var roiGeometry = roi.geometry();

// Quality-controlled building-height samples. Only geometry and Height are
// required; the 59 predictor properties are sampled below from predictorImage.
var heightSamples = ee.FeatureCollection(
  "projects/your-project/assets/training_samples"
);

// ICPIS coding: 0 = non-impervious; 2016 = baseline impervious surface;
// 2017-2023 = expansion first observed in the corresponding year.
var icpis = ee.ImageCollection("projects/your-project/assets/ICPIS")
  .mosaic()
  .select([0]);


// -----------------------------------------------------------------------------
// 2. PLANET NICFI: 4 SPECTRAL + 2 INDICES
// -----------------------------------------------------------------------------

var planetImage = planetCol
  .filterBounds(roi)
  .filterDate(startDate, endDate)
  .median()
  .select(["R", "G", "B", "N"])
  .clip(roiGeometry);

// The index directions follow the supplied original implementation, in which
// NIR is the first band in normalizedDifference.
var planetNDVI = planetImage
  .normalizedDifference(["N", "R"])
  .rename("P_NDVI");
var planetNDWI = planetImage
  .normalizedDifference(["G", "N"])
  .rename("P_NDWI");

var planetFeatures = planetImage.addBands([planetNDVI, planetNDWI]);


// -----------------------------------------------------------------------------
// 3. SENTINEL-1: 2 BACKSCATTER + VVH
// -----------------------------------------------------------------------------

// Supplementary Table 2 specifies an annual mean composite.
var s1Image = s1Col
  .filterBounds(roi)
  .filterDate(startDate, endDate)
  .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
  .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
  .filter(ee.Filter.eq("instrumentMode", "IW"))
  .select(["VV", "VH"])
  .mean()
  .clip(roiGeometry);

// Exact formula used in the supplied original code: VVH = VV * 5^VH.
var vvh = s1Image
  .select("VV")
  .multiply(ee.Image.constant(5).pow(s1Image.select("VH")))
  .rename("VVH");

var sentinel1Features = s1Image.addBands(vvh);


// -----------------------------------------------------------------------------
// 4. SENTINEL-2: 11 SPECTRAL + 7 INDICES
// -----------------------------------------------------------------------------

function maskAndScaleSentinel2(image) {
  var clearMask = image
    .select(cloudScoreBand)
    .gte(clearThreshold);
  var scaledSpectralBands = image.select("B.*").multiply(0.0001);

  return image
    .addBands(scaledSpectralBands, null, true)
    .updateMask(clearMask)
    .copyProperties(image, image.propertyNames());
}

function addSentinel2Indices(image) {
  var ndbi = image.normalizedDifference(["B11", "B8"]).rename("NDBI");
  var rvi64 = image.select("B11").divide(image.select("B4")).rename("RVI64");
  var dvi64 = image.select("B11").subtract(image.select("B4")).rename("DVI64");
  var evi = image.expression(
    "2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))",
    {
      NIR: image.select("B8"),
      RED: image.select("B4"),
      BLUE: image.select("B2")
    }
  ).rename("EVI");
  var lswi = image.normalizedDifference(["B8", "B11"]).rename("LSWI");
  var sei = image.expression(
    "((B1 + B9) - (B3 + B8)) / ((B1 + B9) + (B3 + B8))",
    {
      B1: image.select("B1"),
      B3: image.select("B3"),
      B8: image.select("B8"),
      B9: image.select("B9")
    }
  ).rename("SEI");

  // NDWI is needed internally for CSI but is not a Sentinel-2 output feature
  // in Supplementary Table 2.
  var ndwiForCSI = image.normalizedDifference(["B3", "B8"]);
  var csi = sei.subtract(ndwiForCSI.min(image.select("B8"))).rename("CSI");

  return image.addBands([ndbi, rvi64, dvi64, evi, lswi, sei, csi]);
}

var sentinel2FeatureNames = [
  "B1", "B2", "B3", "B4", "B5", "B6",
  "B7", "B8", "B8A", "B11", "B12",
  "NDBI", "RVI64", "DVI64", "EVI", "LSWI", "SEI", "CSI"
];

var sentinel2Features = s2Col
  .filterBounds(roi)
  .filterDate(startDate, endDate)
  .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 70))
  .linkCollection(s2CloudScoreCol, [cloudScoreBand])
  .map(maskAndScaleSentinel2)
  .map(addSentinel2Indices)
  .select(sentinel2FeatureNames)
  .median()
  .clip(roiGeometry);


// -----------------------------------------------------------------------------
// 5. CNN FINAL-CONVOLUTION FEATURES: 16 + 16
// -----------------------------------------------------------------------------

// Run 03_apply_trained_cnn.js first and export its 32-band cnnFeatures image.
// The Python-to-GEE kernel export performs the training-consistent input
// normalization: RGB / 1600, NIR / 3400 and (VV/VH + 50) / 51.
var cnnFeatures = ee.Image(
  "projects/your-project/assets/cnn_features_" + year
).clip(roiGeometry);

var msDeepNames = [];
var rdDeepNames = [];
for (var deepIndex = 1; deepIndex <= 16; deepIndex++) {
  var suffix = deepIndex < 10 ? "0" + deepIndex : String(deepIndex);
  msDeepNames.push("ms_conv_" + suffix);
  rdDeepNames.push("rd_conv_" + suffix);
}

var planetDeepFeatures = cnnFeatures.select(msDeepNames);
var radarDeepFeatures = cnnFeatures.select(rdDeepNames);


// -----------------------------------------------------------------------------
// 6. ASSEMBLE THE COMPLETE 59-BAND PREDICTOR IMAGE
// -----------------------------------------------------------------------------

var inputFeatures = [
  // Planet NICFI: 22.
  "R", "G", "B", "N", "P_NDVI", "P_NDWI"
].concat(
  msDeepNames,
  // Sentinel-1: 19.
  ["VV", "VH", "VVH"],
  rdDeepNames,
  // Sentinel-2: 18.
  sentinel2FeatureNames
);

var predictorImage = ee.Image.cat([
  planetFeatures,
  planetDeepFeatures,
  sentinel1Features,
  radarDeepFeatures,
  sentinel2Features
]).select(inputFeatures).toFloat().clip(roiGeometry);

print("Expected predictor count", 59);
print("Actual predictor count", predictorImage.bandNames().size());
print("Predictor image", predictorImage);


// -----------------------------------------------------------------------------
// 7. SAMPLE THE PREDICTORS AND SPLIT 80/20
// -----------------------------------------------------------------------------

var modelingSamples = predictorImage.sampleRegions({
  collection: heightSamples,
  properties: [labelProperty],
  scale: scale,
  geometries: true,
  tileScale: 4
}).filter(ee.Filter.notNull(inputFeatures.concat([labelProperty])));

var samplesWithRandom = modelingSamples.randomColumn({
  columnName: "random",
  seed: seed
});

var trainSamples = samplesWithRandom.filter(
  ee.Filter.lt("random", trainingFraction)
);
var validationSamples = samplesWithRandom.filter(
  ee.Filter.gte("random", trainingFraction)
);

print("All valid samples", modelingSamples.size());
print("Training samples", trainSamples.size());
print("Validation samples", validationSamples.size());


// -----------------------------------------------------------------------------
// 8. RF REGRESSION: 100 TREES AND MINIMUM LEAF SIZE 5
// -----------------------------------------------------------------------------

var classifier = ee.Classifier.smileRandomForest({
  numberOfTrees: 100,
  minLeafPopulation: 5,
  bagFraction: 0.5,
  seed: seed
}).setOutputMode("REGRESSION").train({
  features: trainSamples,
  classProperty: labelProperty,
  inputProperties: inputFeatures
});

print("RF model", classifier.explain());
print(
  "Feature importance",
  ee.Dictionary(classifier.explain().get("importance"))
);


// -----------------------------------------------------------------------------
// 9. VALIDATION: RMSE, MAE AND R-SQUARED
// -----------------------------------------------------------------------------

var validated = validationSamples.classify(classifier, "PredictedHeight");

validated = validated.map(function(feature) {
  var observed = ee.Number(feature.get(labelProperty));
  var predicted = ee.Number(feature.get("PredictedHeight"));
  var error = observed.subtract(predicted);
  return feature.set({
    AbsoluteError: error.abs(),
    SquaredError: error.pow(2)
  });
});

var observedMean = ee.Number(validated.aggregate_mean(labelProperty));
validated = validated.map(function(feature) {
  var observed = ee.Number(feature.get(labelProperty));
  return feature.set("SquaredTotal", observed.subtract(observedMean).pow(2));
});

var rmse = ee.Number(validated.aggregate_mean("SquaredError")).sqrt();
var mae = ee.Number(validated.aggregate_mean("AbsoluteError"));
var residualSumOfSquares = ee.Number(validated.aggregate_sum("SquaredError"));
var totalSumOfSquares = ee.Number(validated.aggregate_sum("SquaredTotal"));
var rSquared = ee.Number(1).subtract(
  residualSumOfSquares.divide(totalSumOfSquares)
);

print("Validation RMSE (m)", rmse);
print("Validation MAE (m)", mae);
print("Validation R-squared", rSquared);

Export.table.toDrive({
  collection: validated,
  description: "RF_validation_" + year,
  folder: "Building_height_RF",
  fileNamePrefix: "RF_validation_" + year,
  fileFormat: "CSV"
});


// -----------------------------------------------------------------------------
// 10. BUILDING-HEIGHT INVERSION AND ICPIS MASK
// -----------------------------------------------------------------------------

var regressedImage = predictorImage
  .select(inputFeatures)
  .classify(classifier, "Height")
  .max(0)
  .toFloat();

// Retain impervious surfaces already present by the target year.
var builtSpaceMask = icpis.gt(0).and(icpis.lte(year));
var buildingHeight = regressedImage
  .updateMask(builtSpaceMask)
  .rename("Height_" + year)
  .clip(roiGeometry);

var palette = [
  "081d58", "225ea8", "41b6c4", "c7e9b4", "ffff99",
  "fed976", "ffae42", "fd8d3c", "f03b20", "e31a1c", "b10026"
];

Map.centerObject(roi, 6);
Map.addLayer(
  buildingHeight,
  {min: 0, max: 150, palette: palette},
  "Building height " + year
);


// -----------------------------------------------------------------------------
// 11. EXPORT THE 5 M HEIGHT MAP
// -----------------------------------------------------------------------------

Export.image.toDrive({
  image: buildingHeight,
  description: "Building_height_" + year,
  region: roiGeometry,
  scale: scale,
  folder: "Building_height_RF",
  fileNamePrefix: "Building_height_" + year,
  maxPixels: 1e13,
  fileFormat: "GeoTIFF"
});
