// -----------------------------------------------------------------------------
// STEP 2: EXPORT PLANET NICFI + SENTINEL-1 PATCHES FOR CNN TRAINING
// -----------------------------------------------------------------------------

var points = ee.FeatureCollection(
  "projects/your-project/assets/csgd/sample"
).filter(ee.Filter.notNull(["Height", "year"]));

var planetCol = ee.ImageCollection(
  "projects/planet-nicfi/assets/basemaps/asia"
);
var s1Col = ee.ImageCollection("COPERNICUS/S1_GRD");

var years = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023];
var randomSeed = 2023;
var trainingFraction = 0.80;
var patchSize = 35;
var scale = 5;
var outputCrs = "EPSG:3857";
var batchNum = 5;

// Split once so that a sample cannot occur in both training and validation.
points = points.randomColumn("random", randomSeed);
var samplesTrain = points.filter(ee.Filter.lte("random", trainingFraction));
var samplesVal = points.filter(ee.Filter.gt("random", trainingFraction));

print("all samples", points.size());
print("training samples", samplesTrain.size());
print("validation samples", samplesVal.size());

Map.addLayer(samplesTrain, {color: "FF0000"}, "Training samples", false);
Map.addLayer(samplesVal, {color: "00FF00"}, "Validation samples", false);

var patchList = ee.List.repeat(
  ee.List.repeat(1, patchSize),
  patchSize
);
var patchKernel = ee.Kernel.fixed({
  width: patchSize,
  height: patchSize,
  weights: patchList,
  x: Math.floor(patchSize / 2),
  y: Math.floor(patchSize / 2),
  normalize: false
});

function exportBatches(arraysImage, sampleCollection, splitName, year) {
  var sampleCount = sampleCollection.size().getInfo();

  if (sampleCount === 0) {
    print(splitName + " " + year + ": no samples");
    return;
  }

  var batchSize = Math.ceil(sampleCount / batchNum);
  print(splitName + " " + year + " sample count", sampleCount);
  print(splitName + " " + year + " batch size", batchSize);

  for (var batchIndex = 0; batchIndex < batchNum; batchIndex++) {
    var offset = batchIndex * batchSize;

    if (offset >= sampleCount) {
      break;
    }

    var pointSubset = ee.FeatureCollection(
      sampleCollection.toList(batchSize, offset)
    );

    var patches = arraysImage.sampleRegions({
      collection: pointSubset,
      properties: ["Height", "year"],
      scale: scale,
      tileScale: 8,
      geometries: false
    });

    var taskName =
      "Samples_TF_" + splitName + "_" + year + "_" + batchIndex;

    Export.table.toDrive({
      collection: patches,
      description: taskName,
      fileNamePrefix: taskName,
      folder: "Building_Height_Patches_" + splitName,
      fileFormat: "TFRecord",
      selectors: [
        "R", "G", "B", "NIR", "VV", "VH", "Height", "year"
      ]
    });
  }
}

years.forEach(function(year) {
  // Use only labels belonging to the current year.
  var pointsYear = points.filter(ee.Filter.eq("year", year));
  var yearCount = pointsYear.size().getInfo();
  print("sample size of " + year, yearCount);

  if (yearCount === 0) {
    return;
  }

  var trainYear = samplesTrain.filter(ee.Filter.eq("year", year));
  var valYear = samplesVal.filter(ee.Filter.eq("year", year));
  var sampleRegion = pointsYear.geometry();
  var startDate = ee.Date.fromYMD(year, 1, 1);
  var endDate = startDate.advance(1, "year");

  // The manuscript uses an annual median Planet NICFI composite.
  // Keep the original reflectance values because the local CNN applies its
  // own RGB and NIR normalization.
  var planetImage = planetCol
    .filterBounds(sampleRegion)
    .filterDate(startDate, endDate)
    .median()
    .select(["R", "G", "B", "N"], ["R", "G", "B", "NIR"])
    .resample("bilinear")
    .reproject({crs: outputCrs, scale: scale});

  // The manuscript and Supplementary Table 2 use an annual mean Sentinel-1
  // composite, not a median composite.
  var s1Image = s1Col
    .filterBounds(sampleRegion)
    .filterDate(startDate, endDate)
    .filter(
      ee.Filter.listContains(
        "transmitterReceiverPolarisation",
        "VV"
      )
    )
    .filter(
      ee.Filter.listContains(
        "transmitterReceiverPolarisation",
        "VH"
      )
    )
    .filter(ee.Filter.eq("instrumentMode", "IW"))
    .select(["VV", "VH"])
    .mean()
    .resample("bilinear")
    .reproject({crs: outputCrs, scale: scale});

  var inputImage = ee.Image.cat([planetImage, s1Image])
    .select(["R", "G", "B", "NIR", "VV", "VH"]);
  var arraysImage = inputImage.neighborhoodToArray(patchKernel);

  Map.addLayer(
    planetImage.select(["R", "G", "B"]),
    {min: 0, max: 3000},
    "Planet NICFI " + year,
    false
  );

  exportBatches(arraysImage, trainYear, "Train", year);
  exportBatches(arraysImage, valYear, "Val", year);
});
