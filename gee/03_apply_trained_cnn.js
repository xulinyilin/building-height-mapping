// -----------------------------------------------------------------------------
// 1. USER INPUTS
// -----------------------------------------------------------------------------

// Replace these three example asset IDs with your own objects/assets.
var roi = ee.FeatureCollection("projects/your-project/assets/roi").geometry();

// Four bands in this exact order: R, G, B, NIR.
// The generated multispectral branch performs RGB / 1600 and NIR / 3400.
var planet_im = ee.Image("projects/your-project/assets/planet_annual_image");

// Two bands in this exact order: VV, VH.
// The generated radar branch performs (x - (-50)) / 51.
var radar_im = ee.Image("projects/your-project/assets/sentinel1_annual_image");

// Both input images must have the same grid, projection and nominal scale used
// to prepare the 35 x 35 training patches. Align them before this script if needed.
planet_im = planet_im.select(["R", "G", "B", "NIR"]).clip(roi);
radar_im = radar_im.select(["VV", "VH"]).clip(roi);


// -----------------------------------------------------------------------------
// 2. MULTI-INPUT / MULTI-OUTPUT CONVOLUTION HELPER
// -----------------------------------------------------------------------------

/**
 * Apply one PyTorch-style multi-channel convolution layer.
 * kernelList is structured as [output channel][input channel].
 * Bias is omitted because every trained convolution uses bias = false.
 */
function convNtoN2(inputImage, kernelList, inputChannels, outputChannels, prefix) {
  var outputs = [];

  for (var outputIndex = 0; outputIndex < outputChannels; outputIndex++) {
    var outputImage = ee.Image(0);

    for (var inputIndex = 0; inputIndex < inputChannels; inputIndex++) {
      var oneInput = inputImage.select([inputIndex]);
      var oneResponse = oneInput.convolve(kernelList[outputIndex][inputIndex]);
      outputImage = outputImage.add(oneResponse);
    }

    outputs.push(outputImage.rename(prefix + "_" + outputIndex));
  }

  return ee.Image.cat(outputs);
}


// -----------------------------------------------------------------------------
// 3. TRAINED MULTISPECTRAL BRANCH
// -----------------------------------------------------------------------------

/*
Paste the COMPLETE contents of ms_branch_togee.txt here.

That generated block contains:
  - the correct RGB/NIR normalization;
  - all trained ee.Kernel.fixed definitions;
  - all eight Conv + ReLU operations.

Its final variable is:
  ms_branch_7_out_im
*/

// ... generated multispectral kernel and convolution code omitted ...


// -----------------------------------------------------------------------------
// 4. TRAINED RADAR BRANCH
// -----------------------------------------------------------------------------

/*
Paste the COMPLETE contents of rd_branch_togee.txt here.

That generated block contains:
  - the correct VV/VH normalization;
  - all trained ee.Kernel.fixed definitions;
  - all seven Conv + ReLU operations.

Its final variable is:
  rd_branch_6_out_im
*/

// ... generated radar kernel and convolution code omitted ...


// -----------------------------------------------------------------------------
// 5. FINAL CONVOLUTION OUTPUT: 32 BANDS
// -----------------------------------------------------------------------------

var msFeatureNames = [];
var rdFeatureNames = [];

for (var featureIndex = 1; featureIndex <= 16; featureIndex++) {
  var suffix = featureIndex < 10 ? "0" + featureIndex : String(featureIndex);
  msFeatureNames.push("ms_conv_" + suffix);
  rdFeatureNames.push("rd_conv_" + suffix);
}

var msFinalFeatures = ms_branch_7_out_im.rename(msFeatureNames);
var rdFinalFeatures = rd_branch_6_out_im.rename(rdFeatureNames);

var cnnFeatures = ee.Image.cat([msFinalFeatures, rdFinalFeatures])
  .toFloat()
  .clip(roi);

print("Final CNN feature image", cnnFeatures);
print("CNN feature bands", cnnFeatures.bandNames());
Map.addLayer(
  cnnFeatures.select(["ms_conv_01", "ms_conv_08", "rd_conv_01"]),
  {},
  "Example CNN features",
  false
);


// -----------------------------------------------------------------------------
// 6A. OPTIONAL: EXPORT THE 32-BAND FEATURE IMAGE
// -----------------------------------------------------------------------------

Export.image.toDrive({
  image: cnnFeatures,
  description: "CNN_final_convolution_features",
  folder: "CNN_features",
  fileNamePrefix: "CNN_final_convolution_features",
  region: roi,
  scale: 5,
  maxPixels: 1e13,
  fileFormat: "GeoTIFF"
});


// -----------------------------------------------------------------------------
// 6B. OPTIONAL: EXTRACT FEATURES AT SAMPLE POINTS
// -----------------------------------------------------------------------------

// Replace trainingSamples with your sample FeatureCollection. The Height field
// is retained so the resulting table can be used directly for RF regression.
/*
var featureTable = cnnFeatures.sampleRegions({
  collection: trainingSamples,
  properties: ["sample_id", "year", "Height"],
  scale: 5,
  geometries: true,
  tileScale: 4
});

Export.table.toDrive({
  collection: featureTable,
  description: "CNN_features_at_training_samples",
  folder: "CNN_features",
  fileNamePrefix: "CNN_features_at_training_samples",
  fileFormat: "CSV"
});
*/


// Random-forest regression is implemented separately in
// 04_random_forest_height_mapping.js using the complete 59-band feature set.
